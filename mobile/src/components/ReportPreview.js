/**
 * ReportPreview.js - 제보 미리보기 컴포넌트
 * 제출 전 모든 정보를 확인하는 화면
 */

import React from 'react';
import { View, Text, Image, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';

// 위험 유형 매핑
const HAZARD_TYPES = {
  armed_conflict: { label: '무력충돌', icon: 'conflict', color: '#DC2626' },
  protest_riot: { label: '시위/폭동', icon: 'protest', color: '#F59E0B' },
  checkpoint: { label: '검문소', icon: 'checkpoint', color: '#FF6B6B' },
  road_damage: { label: '도로 손상', icon: 'roadDamage', color: '#F97316' },
  natural_disaster: { label: '자연재해', icon: 'naturalDisaster', color: '#DC2626' },
  other: { label: '기타', icon: 'other', color: '#6B7280' },
};

// 심각도 매핑
const SEVERITY_LABELS = {
  low: { label: '경미', color: '#10B981' },
  medium: { label: '중간', color: '#F59E0B' },
  high: { label: '심각', color: '#DC2626' },
};

export default function ReportPreview({ report, onEdit }) {
  const hazardInfo = HAZARD_TYPES[report.hazardType] || HAZARD_TYPES.other;
  const severityInfo = SEVERITY_LABELS[report.severity] || SEVERITY_LABELS.medium;

  /**
   * 시간을 텍스트로 변환
   */
  const formatTime = (date) => {
    if (!date) return '지금';

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
   * 좌표를 텍스트로 변환
   */
  const formatCoordinates = (lat, lng) => {
    return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <Text style={styles.title}>제보 내용 확인</Text>
      <Text style={styles.subtitle}>
        아래 정보가 정확한지 확인해주세요
      </Text>

      {/* 위험 유형 */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>위험 유형</Text>
          <TouchableOpacity onPress={() => onEdit(1)} activeOpacity={0.7}>
            <Text style={styles.editButton}>수정</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.card}>
          <View style={[styles.iconCircle, { backgroundColor: `${hazardInfo.color}20` }]}>
            <Icon name={hazardInfo.icon} size={24} color={hazardInfo.color} />
          </View>
          <Text style={styles.cardText}>{hazardInfo.label}</Text>
        </View>
      </View>

      {/* 위치 */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>위치</Text>
          <TouchableOpacity onPress={() => onEdit(2)} activeOpacity={0.7}>
            <Text style={styles.editButton}>수정</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.card}>
          <Icon name="location" size={20} color={Colors.primary} />
          <View style={styles.cardTextContainer}>
            <Text style={styles.cardText}>
              {formatCoordinates(report.latitude, report.longitude)}
            </Text>
            {report.accuracy && (
              <Text style={styles.cardSubtext}>
                정확도: ±{Math.round(report.accuracy)}m
              </Text>
            )}
          </View>
        </View>
      </View>

      {/* 심각도 & 시간 */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>상세 정보</Text>
          <TouchableOpacity onPress={() => onEdit(3)} activeOpacity={0.7}>
            <Text style={styles.editButton}>수정</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.card}>
          <View style={styles.infoRow}>
            <Icon name="warning" size={20} color={severityInfo.color} />
            <Text style={styles.cardText}>심각도: </Text>
            <Text style={[styles.cardTextBold, { color: severityInfo.color }]}>
              {severityInfo.label}
            </Text>
          </View>

          <View style={[styles.infoRow, styles.infoRowSpaced]}>
            <Icon name="time" size={20} color={Colors.textSecondary} />
            <Text style={styles.cardText}>발생 시간: </Text>
            <Text style={styles.cardTextBold}>
              {formatTime(report.reportedAt)}
            </Text>
          </View>
        </View>
      </View>

      {/* 사진 */}
      {report.photos && report.photos.length > 0 && (
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>사진 ({report.photos.length})</Text>
            <TouchableOpacity onPress={() => onEdit(3)} activeOpacity={0.7}>
              <Text style={styles.editButton}>수정</Text>
            </TouchableOpacity>
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.photosContainer}>
            {report.photos.map((uri, index) => (
              <Image key={index} source={{ uri }} style={styles.photo} />
            ))}
          </ScrollView>
        </View>
      )}

      {/* 설명 */}
      {report.description && (
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>설명</Text>
            <TouchableOpacity onPress={() => onEdit(3)} activeOpacity={0.7}>
              <Text style={styles.editButton}>수정</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.card}>
            <Text style={styles.descriptionText}>{report.description}</Text>
          </View>
        </View>
      )}

      {/* 조건부 데이터 */}
      {report.conditionalData && Object.keys(report.conditionalData).length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>추가 정보</Text>
          <View style={styles.card}>
            {Object.entries(report.conditionalData).map(([key, value]) => (
              <View key={key} style={styles.conditionalItem}>
                <Text style={styles.conditionalKey}>{key}:</Text>
                <Text style={styles.conditionalValue}>{String(value)}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* 안내 메시지 */}
      <View style={styles.notice}>
        <Icon name="info" size={20} color={Colors.primary} />
        <Text style={styles.noticeText}>
          제보 후 검증을 거쳐 지도에 표시됩니다
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
  title: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
    marginHorizontal: Spacing.lg,
    marginTop: Spacing.lg,
  },
  subtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginBottom: Spacing.lg,
    marginHorizontal: Spacing.lg,
  },
  section: {
    marginBottom: Spacing.lg,
    marginHorizontal: Spacing.lg,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  sectionTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
  },
  editButton: {
    ...Typography.body,
    color: Colors.primary,
    fontWeight: '600',
  },
  card: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  iconCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardTextContainer: {
    flex: 1,
  },
  cardText: {
    ...Typography.body,
    color: Colors.textPrimary,
  },
  cardTextBold: {
    ...Typography.body,
    fontWeight: '600',
    color: Colors.textPrimary,
  },
  cardSubtext: {
    ...Typography.caption,
    color: Colors.textSecondary,
    marginTop: 2,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  infoRowSpaced: {
    marginTop: Spacing.sm,
  },
  photosContainer: {
    marginTop: Spacing.sm,
  },
  photo: {
    width: 120,
    height: 120,
    borderRadius: 8,
    marginRight: Spacing.sm,
    backgroundColor: Colors.surfaceElevated,
  },
  descriptionText: {
    ...Typography.body,
    color: Colors.textPrimary,
    lineHeight: 22,
  },
  conditionalItem: {
    flexDirection: 'row',
    marginBottom: Spacing.xs,
  },
  conditionalKey: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginRight: Spacing.xs,
  },
  conditionalValue: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '500',
  },
  notice: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${Colors.primary}10`,
    padding: Spacing.md,
    borderRadius: 8,
    marginHorizontal: Spacing.lg,
    marginBottom: Spacing.xl,
    gap: Spacing.sm,
  },
  noticeText: {
    ...Typography.body,
    color: Colors.primary,
    flex: 1,
  },
});
