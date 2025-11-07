/**
 * offlineStorage.js - 오프라인 데이터 저장 관리
 * AsyncStorage를 사용하여 오프라인 제보 및 업로드 관리
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

// Storage Keys
const OFFLINE_REPORTS_KEY = '@verisafe:offline_reports';
const PENDING_UPLOADS_KEY = '@verisafe:pending_uploads';

/**
 * 오프라인 제보 저장
 */
export const saveOfflineReport = async (report) => {
  try {
    const reports = await getOfflineReports();
    const timestamp = Date.now();
    const randomStr = Math.random().toString(36).substr(2, 9);
    const newReport = {
      ...report,
      id: `offline_${timestamp}_${randomStr}`,
      offline: true,
      createdAt: new Date().toISOString(),
    };
    reports.push(newReport);
    await AsyncStorage.setItem(OFFLINE_REPORTS_KEY, JSON.stringify(reports));
    return newReport;
  } catch (error) {
    console.error('[OfflineStorage] Failed to save offline report:', error);
    throw error;
  }
};

/**
 * 모든 오프라인 제보 가져오기
 */
export const getOfflineReports = async () => {
  try {
    const data = await AsyncStorage.getItem(OFFLINE_REPORTS_KEY);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error('[OfflineStorage] Failed to get offline reports:', error);
    return [];
  }
};

/**
 * 오프라인 제보 삭제
 */
export const deleteOfflineReport = async (reportId) => {
  try {
    const reports = await getOfflineReports();
    const filtered = reports.filter(r => r.id !== reportId);
    await AsyncStorage.setItem(OFFLINE_REPORTS_KEY, JSON.stringify(filtered));
    return true;
  } catch (error) {
    console.error('[OfflineStorage] Failed to delete offline report:', error);
    return false;
  }
};

/**
 * 업로드 대기 중인 사진 저장
 */
export const savePendingUpload = async (photoUri, reportId) => {
  try {
    const uploads = await getPendingUploads();
    const timestamp = Date.now();
    const randomStr = Math.random().toString(36).substr(2, 9);
    const newUpload = {
      id: `upload_${timestamp}_${randomStr}`,
      photoUri,
      reportId,
      createdAt: new Date().toISOString(),
    };
    uploads.push(newUpload);
    await AsyncStorage.setItem(PENDING_UPLOADS_KEY, JSON.stringify(uploads));
    return newUpload;
  } catch (error) {
    console.error('[OfflineStorage] Failed to save pending upload:', error);
    throw error;
  }
};

/**
 * 업로드 대기 중인 사진 목록 가져오기
 */
export const getPendingUploads = async () => {
  try {
    const data = await AsyncStorage.getItem(PENDING_UPLOADS_KEY);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error('[OfflineStorage] Failed to get pending uploads:', error);
    return [];
  }
};

/**
 * 업로드 완료 후 제거
 */
export const removePendingUpload = async (uploadId) => {
  try {
    const uploads = await getPendingUploads();
    const filtered = uploads.filter(u => u.id !== uploadId);
    await AsyncStorage.setItem(PENDING_UPLOADS_KEY, JSON.stringify(filtered));
    return true;
  } catch (error) {
    console.error('[OfflineStorage] Failed to remove pending upload:', error);
    return false;
  }
};

/**
 * 오프라인 제보 개수 가져오기
 */
export const getOfflineReportsCount = async () => {
  try {
    const reports = await getOfflineReports();
    return reports.length;
  } catch (error) {
    return 0;
  }
};

/**
 * 업로드 대기 개수 가져오기
 */
export const getPendingUploadsCount = async () => {
  try {
    const uploads = await getPendingUploads();
    return uploads.length;
  } catch (error) {
    return 0;
  }
};
