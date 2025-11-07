/**
 * StepIndicator.js - 단계별 진행 표시 컴포넌트
 * 제보 작성 시 현재 단계를 시각적으로 표시
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors, Typography, Spacing } from '../styles';

const STEPS = [
  { id: 1, label: '위험 유형' },
  { id: 2, label: '위치 확인' },
  { id: 3, label: '상세 정보' },
  { id: 4, label: '검토' },
];

export default function StepIndicator({ currentStep }) {
  return (
    <View style={styles.container}>
      <View style={styles.stepsContainer}>
        {STEPS.map((step, index) => {
          const isCompleted = step.id < currentStep;
          const isCurrent = step.id === currentStep;
          const isUpcoming = step.id > currentStep;

          return (
            <React.Fragment key={step.id}>
              {/* Step Circle */}
              <View style={styles.stepItem}>
                <View
                  style={[
                    styles.circle,
                    isCompleted && styles.circleCompleted,
                    isCurrent && styles.circleCurrent,
                    isUpcoming && styles.circleUpcoming,
                  ]}
                >
                  <Text
                    style={[
                      styles.stepNumber,
                      (isCompleted || isCurrent) && styles.stepNumberActive,
                    ]}
                  >
                    {step.id}
                  </Text>
                </View>
                <Text
                  style={[
                    styles.label,
                    isCurrent && styles.labelCurrent,
                    isUpcoming && styles.labelUpcoming,
                  ]}
                >
                  {step.label}
                </Text>
              </View>

              {/* Connector Line */}
              {index < STEPS.length - 1 && (
                <View
                  style={[
                    styles.connector,
                    isCompleted && styles.connectorCompleted,
                  ]}
                />
              )}
            </React.Fragment>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: Spacing.lg,
    paddingHorizontal: Spacing.md,
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  stepsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  stepItem: {
    alignItems: 'center',
    flex: 1,
  },
  circle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xs,
  },
  circleCompleted: {
    backgroundColor: Colors.success,
  },
  circleCurrent: {
    backgroundColor: Colors.primary,
  },
  circleUpcoming: {
    backgroundColor: Colors.surfaceElevated,
    borderWidth: 2,
    borderColor: Colors.border,
  },
  stepNumber: {
    ...Typography.caption,
    fontWeight: '600',
    color: Colors.textTertiary,
  },
  stepNumberActive: {
    color: Colors.textInverse,
  },
  label: {
    ...Typography.caption,
    color: Colors.textSecondary,
    textAlign: 'center',
    fontSize: 11,
  },
  labelCurrent: {
    color: Colors.primary,
    fontWeight: '600',
  },
  labelUpcoming: {
    color: Colors.textTertiary,
  },
  connector: {
    height: 2,
    flex: 1,
    backgroundColor: Colors.border,
    marginHorizontal: -8,
    marginBottom: 24,
  },
  connectorCompleted: {
    backgroundColor: Colors.success,
  },
});
