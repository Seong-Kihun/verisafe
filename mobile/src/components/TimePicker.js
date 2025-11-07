/**
 * TimePicker.js - 발생 시간 선택 컴포넌트
 * 위험이 언제 발생했는지 선택
 */

import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Modal, Platform } from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';

const TIME_PRESETS = [
  { label: '지금 발생', value: 'now', minutes: 0 },
  { label: '1시간 이내', value: '1h', minutes: 60 },
  { label: '오늘', value: 'today', minutes: 240 },
  { label: '어제', value: 'yesterday', minutes: 1440 },
  { label: '직접 입력', value: 'custom', minutes: null },
];

export default function TimePicker({ value, onChange }) {
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState('now');

  /**
   * 프리셋 선택
   */
  const handlePresetSelect = (preset) => {
    setSelectedPreset(preset.value);

    if (preset.value === 'custom') {
      setShowDatePicker(true);
      return;
    }

    const now = new Date();
    const reportedTime = new Date(now.getTime() - preset.minutes * 60000);
    onChange(reportedTime);
  };

  /**
   * 커스텀 날짜/시간 선택
   */
  const handleDateChange = (event, selectedDate) => {
    if (Platform.OS === 'android') {
      setShowDatePicker(false);
    }

    if (selectedDate) {
      onChange(selectedDate);
    }
  };

  /**
   * 선택된 시간을 텍스트로 표시
   */
  const getTimeLabel = () => {
    if (!value) return '지금 발생';

    const now = new Date();
    const diffMinutes = Math.floor((now - value) / 60000);

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

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Icon name="time" size={20} color={Colors.textSecondary} />
        <Text style={styles.title}>언제 발생했나요?</Text>
      </View>

      <View style={styles.presetsContainer}>
        {TIME_PRESETS.map((preset) => {
          const isSelected = selectedPreset === preset.value;

          return (
            <TouchableOpacity
              key={preset.value}
              style={[styles.preset, isSelected && styles.presetSelected]}
              onPress={() => handlePresetSelect(preset)}
              activeOpacity={0.7}
            >
              <Text
                style={[
                  styles.presetLabel,
                  isSelected && styles.presetLabelSelected,
                ]}
              >
                {preset.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {value && (
        <View style={styles.selectedTimeContainer}>
          <Icon name="check-box" size={20} color={Colors.success} />
          <Text style={styles.selectedTimeText}>{getTimeLabel()}</Text>
        </View>
      )}

      {/* Date Picker Modal (iOS) */}
      {Platform.OS === 'ios' && showDatePicker && (
        <Modal
          transparent
          animationType="slide"
          visible={showDatePicker}
          onRequestClose={() => setShowDatePicker(false)}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContainer}>
              <View style={styles.modalHeader}>
                <TouchableOpacity onPress={() => setShowDatePicker(false)}>
                  <Text style={styles.modalButton}>취소</Text>
                </TouchableOpacity>
                <Text style={styles.modalTitle}>시간 선택</Text>
                <TouchableOpacity
                  onPress={() => {
                    setShowDatePicker(false);
                  }}
                >
                  <Text style={[styles.modalButton, styles.modalButtonDone]}>
                    완료
                  </Text>
                </TouchableOpacity>
              </View>

              <DateTimePicker
                value={value || new Date()}
                mode="datetime"
                display="spinner"
                onChange={handleDateChange}
                maximumDate={new Date()}
                locale="ko-KR"
              />
            </View>
          </View>
        </Modal>
      )}

      {/* Date Picker (Android) */}
      {Platform.OS === 'android' && showDatePicker && (
        <DateTimePicker
          value={value || new Date()}
          mode="datetime"
          display="default"
          onChange={handleDateChange}
          maximumDate={new Date()}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: Spacing.md,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.sm,
    gap: Spacing.xs,
  },
  title: {
    ...Typography.body,
    fontWeight: '600',
    color: Colors.textPrimary,
  },
  presetsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  preset: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 20,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  presetSelected: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
  },
  presetLabel: {
    ...Typography.caption,
    color: Colors.textSecondary,
    fontWeight: '500',
  },
  presetLabelSelected: {
    color: Colors.textInverse,
    fontWeight: '600',
  },
  selectedTimeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: Spacing.md,
    padding: Spacing.sm,
    backgroundColor: `${Colors.success}10`,
    borderRadius: 8,
    gap: Spacing.xs,
  },
  selectedTimeText: {
    ...Typography.body,
    color: Colors.success,
    fontWeight: '600',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContainer: {
    backgroundColor: Colors.surface,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingBottom: 34, // iPhone X bottom safe area
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  modalTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
  },
  modalButton: {
    ...Typography.body,
    color: Colors.primary,
  },
  modalButtonDone: {
    fontWeight: '600',
  },
});
