/**
 * ReportTracking.js - 제보 추적 및 영향 통계
 * 사용자의 제보 상태와 영향력을 표시
 */

import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';

const STATUS_INFO = {
  pending: {
    label: '검토 중',
    color: Colors.warning,
    icon: 'time',
    description: '관리자 검토 대기 중입니다',
  },
  verified: {
    label: '검증됨',
    color: Colors.success,
    icon: 'check-box',
    description: '검증이 완료되어 지도에 표시됩니다',
  },
  rejected: {
    label: '반려됨',
    color: Colors.danger,
    icon: 'close',
    description: '검증에 실패했습니다',
  },
};

export default function ReportTracking({ report, onClose }) {
  const statusInfo = STATUS_INFO[report.status] || STATUS_INFO.pending;

  /**
   * 시간 포맷팅
   */
  const formatTime = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMinutes = Math.floor((now - date) / 60000);

    if (diffMinutes < 60) {
      return `${diffMinutes}분 전`;
    } else if (diffMinutes < 1440) {
      const hours = Math.floor(diffMinutes / 60);
      return `${hours}시간 전`;
    } else {
      const days = Math.floor(diffMinutes / 1440);
      if (days === 1) return '어제';
      return `${days}일 전`;
    }
  };

  /**
   * 날짜 포맷팅
   */
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');

    return `${year}.${month}.${day} ${hours}:${minutes}`;
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* 헤더 */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.title}>제보 추적</Text>
          <Text style={styles.reportId}>ID: {report.id?.slice(0, 8) || 'N/A'}</Text>
        </View>
        {onClose && (
          <TouchableOpacity onPress={onClose} style={styles.closeButton}>
            <Icon name="close" size={24} color={Colors.textSecondary} />
          </TouchableOpacity>
        )}
      </View>

      {/* 상태 카드 */}
      <View style={[styles.statusCard, { borderColor: statusInfo.color }]}>
        <View style={[styles.statusIcon, { backgroundColor: `${statusInfo.color}20` }]}>
          <Icon name={statusInfo.icon} size={32} color={statusInfo.color} />
        </View>
        <Text style={[styles.statusLabel, { color: statusInfo.color }]}>
          {statusInfo.label}
        </Text>
        <Text style={styles.statusDescription}>{statusInfo.description}</Text>
      </View>

      {/* 영향 통계 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>커뮤니티 영향</Text>

        <View style={styles.impactGrid}>
          <View style={styles.impactCard}>
            <Icon name="person" size={32} color={Colors.primary} />
            <Text style={styles.impactValue}>{report.impact_count || 0}</Text>
            <Text style={styles.impactLabel}>도움받은 사용자</Text>
          </View>

          <View style={styles.impactCard}>
            <Icon name="route" size={32} color={Colors.success} />
            <Text style={styles.impactValue}>
              {report.routes_affected || 0}
            </Text>
            <Text style={styles.impactLabel}>경로에 반영</Text>
          </View>
        </View>

        {report.impact_count > 0 && (
          <View style={styles.achievementBadge}>
            <Icon name="star" size={20} color={Colors.warning} />
            <Text style={styles.achievementText}>
              {report.impact_count >= 50
                ? '🏆 슈퍼 제보자'
                : report.impact_count >= 20
                ? '⭐ 활동적인 제보자'
                : '👍 감사합니다'}
            </Text>
          </View>
        )}
      </View>

      {/* 타임라인 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>진행 상황</Text>

        <View style={styles.timeline}>
          {/* 제보 생성 */}
          <View style={styles.timelineItem}>
            <View style={styles.timelineDot} />
            <View style={styles.timelineContent}>
              <View style={styles.timelineHeader}>
                <Text style={styles.timelineTitle}>제보 생성</Text>
                <Text style={styles.timelineTime}>
                  {formatTime(report.created_at)}
                </Text>
              </View>
              <Text style={styles.timelineDescription}>
                {formatDate(report.created_at)}
              </Text>
            </View>
          </View>

          {/* 검증 완료 */}
          {report.verified_at && (
            <View style={styles.timelineItem}>
              <View
                style={[
                  styles.timelineDot,
                  { backgroundColor: Colors.success },
                ]}
              />
              <View style={styles.timelineContent}>
                <View style={styles.timelineHeader}>
                  <Text style={styles.timelineTitle}>검증 완료</Text>
                  <Text style={styles.timelineTime}>
                    {formatTime(report.verified_at)}
                  </Text>
                </View>
                <Text style={styles.timelineDescription}>
                  {formatDate(report.verified_at)}
                </Text>
              </View>
            </View>
          )}

          {/* 지도에 표시 */}
          {report.status === 'verified' && (
            <View style={styles.timelineItem}>
              <View
                style={[
                  styles.timelineDot,
                  { backgroundColor: Colors.primary },
                ]}
              />
              <View style={styles.timelineContent}>
                <View style={styles.timelineHeader}>
                  <Text style={styles.timelineTitle}>지도에 표시됨</Text>
                  <Text style={styles.timelineTime}>현재</Text>
                </View>
                <Text style={styles.timelineDescription}>
                  다른 사용자들이 확인할 수 있습니다
                </Text>
              </View>
            </View>
          )}
        </View>
      </View>

      {/* 제보 상세 정보 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>제보 정보</Text>

        <View style={styles.detailsCard}>
          <View style={styles.detailRow}>
            <Icon name="location" size={18} color={Colors.textSecondary} />
            <Text style={styles.detailLabel}>위치:</Text>
            <Text style={styles.detailValue}>
              {report.latitude?.toFixed(5)}, {report.longitude?.toFixed(5)}
            </Text>
          </View>

          {report.severity && (
            <View style={styles.detailRow}>
              <Icon name="warning" size={18} color={Colors.textSecondary} />
              <Text style={styles.detailLabel}>심각도:</Text>
              <Text style={styles.detailValue}>
                {report.severity === 'high'
                  ? '심각'
                  : report.severity === 'medium'
                  ? '중간'
                  : '경미'}
              </Text>
            </View>
          )}

          {report.hazard_type && (
            <View style={styles.detailRow}>
              <Icon name="info" size={18} color={Colors.textSecondary} />
              <Text style={styles.detailLabel}>유형:</Text>
              <Text style={styles.detailValue}>{report.hazard_type}</Text>
            </View>
          )}
        </View>
      </View>

      {/* 푸터 정보 */}
      <View style={styles.footer}>
        <Icon name="info" size={16} color={Colors.textTertiary} />
        <Text style={styles.footerText}>
          제보 상태는 실시간으로 업데이트됩니다
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    padding: Spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  headerLeft: {
    flex: 1,
  },
  title: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  reportId: {
    ...Typography.caption,
    color: Colors.textTertiary,
    fontFamily: 'monospace',
  },
  closeButton: {
    padding: Spacing.sm,
  },
  statusCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: Spacing.xl,
    margin: Spacing.lg,
    alignItems: 'center',
    borderWidth: 2,
  },
  statusIcon: {
    width: 72,
    height: 72,
    borderRadius: 36,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.md,
  },
  statusLabel: {
    ...Typography.h2,
    marginBottom: Spacing.xs,
  },
  statusDescription: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  section: {
    padding: Spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  sectionTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.md,
  },
  impactGrid: {
    flexDirection: 'row',
    gap: Spacing.md,
  },
  impactCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    alignItems: 'center',
  },
  impactValue: {
    ...Typography.h1,
    color: Colors.textPrimary,
    marginTop: Spacing.sm,
    marginBottom: Spacing.xs,
  },
  impactLabel: {
    ...Typography.caption,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  achievementBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${Colors.warning}10`,
    padding: Spacing.md,
    borderRadius: 12,
    marginTop: Spacing.md,
    gap: Spacing.sm,
  },
  achievementText: {
    ...Typography.body,
    color: Colors.warning,
    fontWeight: '600',
    flex: 1,
  },
  timeline: {
    gap: Spacing.lg,
  },
  timelineItem: {
    flexDirection: 'row',
    gap: Spacing.md,
  },
  timelineDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: Colors.primary,
    marginTop: 6,
  },
  timelineContent: {
    flex: 1,
  },
  timelineHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.xs,
  },
  timelineTitle: {
    ...Typography.body,
    fontWeight: '600',
    color: Colors.textPrimary,
  },
  timelineTime: {
    ...Typography.caption,
    color: Colors.textTertiary,
  },
  timelineDescription: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  detailsCard: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    gap: Spacing.md,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  detailLabel: {
    ...Typography.body,
    color: Colors.textSecondary,
    width: 60,
  },
  detailValue: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '500',
    flex: 1,
  },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.lg,
    gap: Spacing.sm,
  },
  footerText: {
    ...Typography.caption,
    color: Colors.textTertiary,
    flex: 1,
  },
});
