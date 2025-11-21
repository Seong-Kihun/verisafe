/**
 * syncService.js - 오프라인 제보 동기화 서비스
 * 네트워크 연결 시 자동으로 오프라인 제보를 서버로 전송
 */

import { reportAPI } from './api';
import api from './api';
import {
  getOfflineReports,
  deleteOfflineReport,
  getPendingUploads,
  removePendingUpload,
} from '../utils/offlineStorage';
import { isOnline } from '../utils/networkUtils';

/**
 * 오프라인 제보를 서버로 동기화
 */
export const syncOfflineReports = async () => {
  try {
    // 네트워크 연결 확인
    const online = await isOnline();
    if (!online) {
      if (__DEV__) console.log('[Sync] No internet connection, skipping sync');
      return { success: false, reason: 'offline' };
    }

    // 오프라인 제보 가져오기
    const offlineReports = await getOfflineReports();
    if (offlineReports.length === 0) {
      if (__DEV__) console.log('[Sync] No offline reports to sync');
      return { success: true, synced: 0 };
    }

    if (__DEV__) console.log(`[Sync] Syncing ${offlineReports.length} offline reports`);

    // 각 제보를 병렬로 서버에 전송 (성능 최적화)
    const syncPromises = offlineReports.map(async (report) => {
      try {
        // offline 플래그와 로컬 ID 제거
        const { offline, id, ...reportData } = report;

        // 서버로 전송
        await reportAPI.create(reportData);

        // 성공하면 로컬에서 삭제
        await deleteOfflineReport(report.id);

        if (__DEV__) console.log(`[Sync] Successfully synced report: ${report.id}`);
        return { success: true, report };
      } catch (error) {
        console.error(`[Sync] Failed to sync report ${report.id}:`, error);
        return { success: false, report, error: error.message };
      }
    });

    const results = await Promise.allSettled(syncPromises);

    const successCount = results.filter(r => r.status === 'fulfilled' && r.value.success).length;
    const failedReports = results
      .filter(r => r.status === 'fulfilled' && !r.value.success)
      .map(r => r.value);

    if (__DEV__) console.log(`[Sync] Complete: ${successCount}/${offlineReports.length} successful`);

    return {
      success: true,
      synced: successCount,
      failed: failedReports.length,
      failedReports,
    };
  } catch (error) {
    console.error('[Sync] Sync failed:', error);
    return { success: false, error: error.message };
  }
};

/**
 * 업로드 대기 중인 사진 동기화
 */
export const syncPendingUploads = async () => {
  try {
    const online = await isOnline();
    if (!online) {
      return { success: false, reason: 'offline' };
    }

    const pendingUploads = await getPendingUploads();
    if (pendingUploads.length === 0) {
      return { success: true, synced: 0 };
    }

    if (__DEV__) console.log(`[Sync] Syncing ${pendingUploads.length} pending uploads`);

    let successCount = 0;

    for (const upload of pendingUploads) {
      try {
        if (upload.photoUri) {
          await uploadPhoto(upload.photoUri, upload.reportId);
        }

        await removePendingUpload(upload.id);
        successCount++;
      } catch (error) {
        console.error(`[Sync] Failed to upload ${upload.id}:`, error);
      }
    }

    return { success: true, synced: successCount };
  } catch (error) {
    console.error('[Sync] Upload sync failed:', error);
    return { success: false, error: error.message };
  }
};

/**
 * 전체 동기화 (제보 + 사진)
 */
export const syncAll = async () => {
  try {
    const reportsResult = await syncOfflineReports();
    const uploadsResult = await syncPendingUploads();

    return {
      success: reportsResult.success && uploadsResult.success,
      reports: reportsResult,
      uploads: uploadsResult,
    };
  } catch (error) {
    console.error('[Sync] Full sync failed:', error);
    return { success: false, error: error.message };
  }
};

/**
 * 사진 업로드 함수
 * @param {string} photoUri - 사진 URI (file://)
 * @param {string} reportId - 제보 ID
 */
const uploadPhoto = async (photoUri, reportId) => {
  try {
    // FormData 생성
    const formData = new FormData();

    // 파일 정보 생성
    const filename = photoUri.split('/').pop();
    const fileType = filename.split('.').pop();
    const mimeType = `image/${fileType === 'jpg' ? 'jpeg' : fileType}`;

    formData.append('photo', {
      uri: photoUri,
      type: mimeType,
      name: filename,
    });

    formData.append('report_id', reportId);

    // 사진 업로드 API 호출
    const response = await api.post('/api/reports/upload-photo', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (__DEV__) console.log('[SyncService] Photo uploaded successfully:', response.data);
    return response.data;
  } catch (error) {
    console.error('[SyncService] Photo upload failed:', error);
    throw error;
  }
};
