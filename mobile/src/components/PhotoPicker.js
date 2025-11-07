/**
 * PhotoPicker.js - 사진 촬영/선택 컴포넌트
 * 카메라로 촬영하거나 갤러리에서 선택
 * 최대 5장까지 첨부 가능
 */

import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Image, StyleSheet, ScrollView, Alert, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';

const MAX_PHOTOS = 5;

export default function PhotoPicker({ photos, onChange }) {
  const [permissionsAsked, setPermissionsAsked] = useState(false);

  /**
   * 카메라 권한 요청
   */
  const requestCameraPermission = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          '권한 필요',
          '사진을 촬영하려면 카메라 권한이 필요합니다.',
          [{ text: '확인' }]
        );
        return false;
      }
      return true;
    } catch (error) {
      console.error('Camera permission error:', error);
      return false;
    }
  };

  /**
   * 갤러리 권한 요청
   */
  const requestMediaLibraryPermission = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          '권한 필요',
          '사진을 선택하려면 갤러리 접근 권한이 필요합니다.',
          [{ text: '확인' }]
        );
        return false;
      }
      return true;
    } catch (error) {
      console.error('Media library permission error:', error);
      return false;
    }
  };

  /**
   * 카메라로 사진 촬영
   */
  const takePhoto = async () => {
    if (photos.length >= MAX_PHOTOS) {
      Alert.alert('최대 개수', `최대 ${MAX_PHOTOS}장까지 첨부할 수 있습니다.`);
      return;
    }

    const hasPermission = await requestCameraPermission();
    if (!hasPermission) return;

    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
        exif: false, // 익명성 보호: 위치 메타데이터 제거
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const newPhotos = [...photos, result.assets[0].uri];
        onChange(newPhotos);
      }
    } catch (error) {
      console.error('Error taking photo:', error);
      Alert.alert('오류', '사진 촬영 중 오류가 발생했습니다.');
    }
  };

  /**
   * 갤러리에서 사진 선택
   */
  const pickPhoto = async () => {
    if (photos.length >= MAX_PHOTOS) {
      Alert.alert('최대 개수', `최대 ${MAX_PHOTOS}장까지 첨부할 수 있습니다.`);
      return;
    }

    const hasPermission = await requestMediaLibraryPermission();
    if (!hasPermission) return;

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
        exif: false, // 익명성 보호: 위치 메타데이터 제거
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const newPhotos = [...photos, result.assets[0].uri];
        onChange(newPhotos);
      }
    } catch (error) {
      console.error('Error picking photo:', error);
      Alert.alert('오류', '사진 선택 중 오류가 발생했습니다.');
    }
  };

  /**
   * 사진 삭제
   */
  const removePhoto = (index) => {
    const newPhotos = photos.filter((_, i) => i !== index);
    onChange(newPhotos);
  };

  /**
   * 선택 옵션 표시
   */
  const showOptions = () => {
    Alert.alert(
      '사진 추가',
      '사진을 어떻게 추가하시겠습니까?',
      [
        { text: '카메라로 촬영', onPress: takePhoto },
        { text: '갤러리에서 선택', onPress: pickPhoto },
        { text: '취소', style: 'cancel' },
      ],
      { cancelable: true }
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Icon name="camera" size={20} color={Colors.textSecondary} />
        <Text style={styles.title}>사진 추가 (선택)</Text>
        <Text style={styles.count}>{photos.length}/{MAX_PHOTOS}</Text>
      </View>

      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.scrollView}>
        {photos.map((uri, index) => (
          <View key={index} style={styles.photoContainer}>
            <Image source={{ uri }} style={styles.photo} />
            <TouchableOpacity
              style={styles.removeButton}
              onPress={() => removePhoto(index)}
              activeOpacity={0.8}
            >
              <Icon name="close" size={16} color={Colors.textInverse} />
            </TouchableOpacity>
          </View>
        ))}

        {photos.length < MAX_PHOTOS && (
          <TouchableOpacity
            style={styles.addButton}
            onPress={showOptions}
            activeOpacity={0.7}
          >
            <Icon name="camera" size={32} color={Colors.textTertiary} />
            <Text style={styles.addButtonText}>사진 추가</Text>
          </TouchableOpacity>
        )}
      </ScrollView>

      {photos.length === 0 && (
        <Text style={styles.hint}>
          사진을 추가하면 제보의 신뢰도가 높아집니다
        </Text>
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
    color: Colors.textPrimary,
    flex: 1,
  },
  count: {
    ...Typography.caption,
    color: Colors.textTertiary,
  },
  scrollView: {
    marginVertical: Spacing.sm,
  },
  photoContainer: {
    position: 'relative',
    marginRight: Spacing.sm,
  },
  photo: {
    width: 100,
    height: 100,
    borderRadius: 8,
    backgroundColor: Colors.surfaceElevated,
  },
  removeButton: {
    position: 'absolute',
    top: 4,
    right: 4,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: 12,
    width: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  addButton: {
    width: 100,
    height: 100,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: Colors.border,
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.surface,
  },
  addButtonText: {
    ...Typography.caption,
    color: Colors.textTertiary,
    marginTop: Spacing.xs,
  },
  hint: {
    ...Typography.caption,
    color: Colors.textTertiary,
    marginTop: Spacing.xs,
    textAlign: 'center',
  },
});
