# Phase 1 구현 계획: 생존 핵심 기능

## 목표
VeriSafe의 가장 중요한 차별화 요소인 "생존 중심 안전 기능"을 2주 내에 구현하여 상용 네비게이션 앱과 확실히 차별화된 가치 제공.

## 구현 범위 (5개 핵심 기능)

### 1. 긴급 SOS 버튼 ⭐ (Priority 1)
**예상 시간**: 1일
**난이도**: 중
**의존성**: Emergency contacts, Safe havens

### 2. 위험 지역 진입 경고 ⭐ (Priority 2)
**예상 시간**: 1일
**난이도**: 하
**의존성**: Risk assessment API, Geofencing

### 3. 안전 체크인 시스템 ⭐ (Priority 3)
**예상 시간**: 2일
**난이도**: 중상
**의존성**: Emergency contacts, Push notifications

### 4. 긴급 연락망 ⭐ (Priority 4)
**예상 시간**: 1일
**난이도**: 하
**의존성**: Contact picker, SMS/Push API

### 5. 안전 대피처 표시 ⭐ (Priority 5)
**예상 시간**: 1일
**난이도**: 하
**의존성**: POI database, Map markers

---

## 구현 순서 및 타임라인

### Day 1-2: 긴급 연락망 (Foundation)
다른 모든 기능의 기반이 되는 긴급 연락망을 먼저 구축

### Day 3-4: 안전 대피처 표시
지도에 안전한 장소를 표시하여 사용자가 피난처를 파악

### Day 5-6: 긴급 SOS 버튼
연락망과 대피처를 활용하는 핵심 긴급 기능

### Day 7-8: 위험 지역 진입 경고
실시간으로 위험을 감지하고 경고

### Day 9-12: 안전 체크인 시스템
자동화된 안전 확인 메커니즘

### Day 13-14: 통합 테스트 및 버그 수정

---

## 1. 긴급 연락망 (Emergency Contacts)

### 1.1 데이터 구조

#### Storage Schema
```javascript
// mobile/src/services/storage.js - 추가할 부분

export const emergencyContactsStorage = {
  STORAGE_KEY: '@verisafe_emergency_contacts',

  async getAll() {
    try {
      const data = await AsyncStorage.getItem(this.STORAGE_KEY);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Failed to get emergency contacts:', error);
      return [];
    }
  },

  async add(contact) {
    try {
      const contacts = await this.getAll();

      // 최대 5명 제한
      if (contacts.length >= 5) {
        throw new Error('Maximum 5 emergency contacts allowed');
      }

      const newContact = {
        id: `contact_${Date.now()}`,
        name: contact.name,
        phone: contact.phone,
        email: contact.email || null,
        relationship: contact.relationship || 'other', // family, friend, colleague, other
        priority: contact.priority || contacts.length + 1, // 1-5
        shareLocation: contact.shareLocation ?? true, // 위치 공유 동의
        createdAt: new Date().toISOString(),
      };

      contacts.push(newContact);
      await AsyncStorage.setItem(this.STORAGE_KEY, JSON.stringify(contacts));
      return newContact;
    } catch (error) {
      console.error('Failed to add emergency contact:', error);
      return null;
    }
  },

  async update(contactId, updates) {
    try {
      const contacts = await this.getAll();
      const index = contacts.findIndex(c => c.id === contactId);

      if (index === -1) return false;

      contacts[index] = { ...contacts[index], ...updates };
      await AsyncStorage.setItem(this.STORAGE_KEY, JSON.stringify(contacts));
      return true;
    } catch (error) {
      console.error('Failed to update emergency contact:', error);
      return false;
    }
  },

  async remove(contactId) {
    try {
      const contacts = await this.getAll();
      const filtered = contacts.filter(c => c.id !== contactId);
      await AsyncStorage.setItem(this.STORAGE_KEY, JSON.stringify(filtered));
      return true;
    } catch (error) {
      console.error('Failed to remove emergency contact:', error);
      return false;
    }
  },

  async reorder(contactIds) {
    try {
      const contacts = await this.getAll();
      const reordered = contactIds.map((id, index) => {
        const contact = contacts.find(c => c.id === id);
        return { ...contact, priority: index + 1 };
      });
      await AsyncStorage.setItem(this.STORAGE_KEY, JSON.stringify(reordered));
      return true;
    } catch (error) {
      console.error('Failed to reorder emergency contacts:', error);
      return false;
    }
  },
};
```

### 1.2 UI Components

#### EmergencyContactsScreen.js
새 파일: `mobile/src/screens/EmergencyContactsScreen.js`

```javascript
/**
 * 긴급 연락망 관리 화면
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Linking,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Colors, Typography, Spacing } from '../styles';
import { emergencyContactsStorage } from '../services/storage';
import Icon from '../components/icons/Icon';

export default function EmergencyContactsScreen({ navigation }) {
  const [loading, setLoading] = useState(true);
  const [contacts, setContacts] = useState([]);

  useFocusEffect(
    useCallback(() => {
      loadContacts();
    }, [])
  );

  const loadContacts = async () => {
    try {
      const data = await emergencyContactsStorage.getAll();
      // 우선순위 순으로 정렬
      const sorted = data.sort((a, b) => a.priority - b.priority);
      setContacts(sorted);
    } catch (error) {
      console.error('Failed to load emergency contacts:', error);
      Alert.alert('오류', '긴급 연락망을 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = (contactId, contactName) => {
    Alert.alert(
      '삭제 확인',
      `${contactName}을(를) 긴급 연락망에서 삭제하시겠습니까?`,
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            const success = await emergencyContactsStorage.remove(contactId);
            if (success) {
              setContacts(prev => prev.filter(c => c.id !== contactId));
            } else {
              Alert.alert('오류', '삭제에 실패했습니다.');
            }
          },
        },
      ]
    );
  };

  const handleCall = (phone) => {
    Linking.openURL(`tel:${phone}`);
  };

  const getRelationshipLabel = (relationship) => {
    const labels = {
      family: '가족',
      friend: '친구',
      colleague: '동료',
      other: '기타',
    };
    return labels[relationship] || '기타';
  };

  const getRelationshipIcon = (relationship) => {
    const icons = {
      family: 'people',
      friend: 'person',
      colleague: 'work',
      other: 'contactPhone',
    };
    return icons[relationship] || 'contactPhone';
  };

  const renderItem = ({ item, index }) => (
    <View style={styles.contactCard}>
      <View style={styles.contactHeader}>
        <View style={styles.priorityBadge}>
          <Text style={styles.priorityText}>{item.priority}</Text>
        </View>
        <View style={styles.contactInfo}>
          <Text style={styles.contactName}>{item.name}</Text>
          <Text style={styles.contactPhone}>{item.phone}</Text>
          {item.email && (
            <Text style={styles.contactEmail}>{item.email}</Text>
          )}
        </View>
        <View style={styles.relationshipBadge}>
          <Icon
            name={getRelationshipIcon(item.relationship)}
            size={16}
            color={Colors.primary}
          />
          <Text style={styles.relationshipText}>
            {getRelationshipLabel(item.relationship)}
          </Text>
        </View>
      </View>

      <View style={styles.contactMeta}>
        <View style={styles.metaItem}>
          <Icon
            name={item.shareLocation ? 'locationOn' : 'locationOff'}
            size={16}
            color={item.shareLocation ? Colors.success : Colors.textTertiary}
          />
          <Text style={[
            styles.metaText,
            item.shareLocation ? styles.metaActive : styles.metaInactive
          ]}>
            {item.shareLocation ? '위치 공유 ON' : '위치 공유 OFF'}
          </Text>
        </View>
      </View>

      <View style={styles.contactActions}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => handleCall(item.phone)}
        >
          <Icon name="phone" size={20} color={Colors.success} />
          <Text style={[styles.actionButtonText, { color: Colors.success }]}>
            전화
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => navigation.navigate('EmergencyContactEdit', { contact: item })}
        >
          <Icon name="edit" size={20} color={Colors.primary} />
          <Text style={styles.actionButtonText}>수정</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.deleteButton]}
          onPress={() => handleDelete(item.id, item.name)}
        >
          <Icon name="delete" size={20} color={Colors.danger} />
          <Text style={[styles.actionButtonText, styles.deleteText]}>삭제</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {contacts.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Icon name="contactPhone" size={64} color={Colors.textTertiary} />
          <Text style={styles.emptyTitle}>긴급 연락망이 비어있습니다</Text>
          <Text style={styles.emptyText}>
            위급한 상황에서 자동으로{'\n'}
            연락을 받을 사람을 추가하세요{'\n'}
            (최대 5명)
          </Text>
          <TouchableOpacity
            style={styles.addFirstButton}
            onPress={() => navigation.navigate('EmergencyContactEdit')}
          >
            <Icon name="add" size={20} color={Colors.textInverse} />
            <Text style={styles.addFirstButtonText}>첫 연락처 추가</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <Text style={styles.headerTitle}>긴급 연락망 {contacts.length}/5</Text>
              <Text style={styles.headerSubtitle}>
                우선순위 순으로 알림이 전송됩니다
              </Text>
            </View>
            {contacts.length < 5 && (
              <TouchableOpacity
                style={styles.addButton}
                onPress={() => navigation.navigate('EmergencyContactEdit')}
              >
                <Icon name="add" size={24} color={Colors.primary} />
              </TouchableOpacity>
            )}
          </View>

          <FlatList
            data={contacts}
            renderItem={renderItem}
            keyExtractor={item => item.id}
            contentContainerStyle={styles.listContainer}
          />

          <View style={styles.infoBox}>
            <Icon name="info" size={20} color={Colors.warning} />
            <Text style={styles.infoText}>
              SOS 버튼을 누르거나 안전 체크인을 놓치면{'\n'}
              등록된 연락처로 자동 알림이 전송됩니다
            </Text>
          </View>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Colors.background,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: Spacing.xl,
    backgroundColor: Colors.background,
  },
  emptyTitle: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginTop: Spacing.lg,
    marginBottom: Spacing.sm,
  },
  emptyText: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: Spacing.xl,
    lineHeight: 24,
  },
  addFirstButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.primary,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderRadius: 12,
  },
  addFirstButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
    fontWeight: '600',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: Spacing.lg,
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  headerLeft: {
    flex: 1,
  },
  headerTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  headerSubtitle: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  addButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.primaryLight + '20',
    justifyContent: 'center',
    alignItems: 'center',
  },
  listContainer: {
    padding: Spacing.md,
  },
  contactCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  contactHeader: {
    flexDirection: 'row',
    marginBottom: Spacing.sm,
    alignItems: 'flex-start',
  },
  priorityBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  priorityText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: '700',
  },
  contactInfo: {
    flex: 1,
  },
  contactName: {
    ...Typography.h4,
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  contactPhone: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginBottom: 2,
  },
  contactEmail: {
    ...Typography.bodySmall,
    color: Colors.textTertiary,
  },
  relationshipBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: Colors.primaryLight + '20',
    paddingHorizontal: Spacing.sm,
    paddingVertical: 4,
    borderRadius: 8,
  },
  relationshipText: {
    ...Typography.captionSmall,
    color: Colors.primary,
    fontWeight: '600',
  },
  contactMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.md,
    paddingLeft: 32 + Spacing.md,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  metaText: {
    ...Typography.captionSmall,
  },
  metaActive: {
    color: Colors.success,
    fontWeight: '600',
  },
  metaInactive: {
    color: Colors.textTertiary,
  },
  contactActions: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.primaryLight + '20',
    paddingVertical: Spacing.sm,
    borderRadius: 12,
  },
  actionButtonText: {
    ...Typography.labelSmall,
    color: Colors.primary,
    fontWeight: '600',
  },
  deleteButton: {
    backgroundColor: Colors.danger + '10',
  },
  deleteText: {
    color: Colors.danger,
  },
  infoBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    backgroundColor: Colors.warning + '10',
    padding: Spacing.md,
    margin: Spacing.lg,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.warning + '40',
  },
  infoText: {
    ...Typography.captionSmall,
    color: Colors.textSecondary,
    flex: 1,
    lineHeight: 18,
  },
});
```

### 1.3 Navigation 추가

`mobile/src/navigation/ProfileStack.js` 수정:

```javascript
// 기존 imports에 추가
import EmergencyContactsScreen from '../screens/EmergencyContactsScreen';
import EmergencyContactEditScreen from '../screens/EmergencyContactEditScreen';

// Stack.Screen 추가
<Stack.Screen
  name="EmergencyContacts"
  component={EmergencyContactsScreen}
  options={{ title: '긴급 연락망' }}
/>
<Stack.Screen
  name="EmergencyContactEdit"
  component={EmergencyContactEditScreen}
  options={{ title: '긴급 연락처 편집' }}
/>
```

### 1.4 ProfileTabScreen 메뉴 추가

`mobile/src/screens/ProfileTabScreen.js`의 "나의 활동" 섹션에 추가:

```javascript
<TouchableOpacity
  style={styles.menuItem}
  onPress={() => navigation.navigate('EmergencyContacts')}
>
  <View style={styles.menuLeft}>
    <Icon name="contactPhone" size={24} color={Colors.danger} />
    <View>
      <Text style={styles.menuLabel}>긴급 연락망</Text>
      <Text style={styles.menuDescription}>위급 상황 자동 알림</Text>
    </View>
  </View>
  <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
</TouchableOpacity>
```

---

## 2. 안전 대피처 표시 (Safe Havens)

### 2.1 데이터베이스 스키마

#### Backend API Endpoint
`backend/routes/safeHavens.js` (새 파일)

```javascript
const express = require('express');
const router = express.Router();
const pool = require('../db');

/**
 * GET /api/safe-havens
 * Query params:
 *   - lat: latitude
 *   - lon: longitude
 *   - radius: search radius in meters (default: 5000)
 *   - category: embassy, hospital, un, police, hotel, shelter (optional)
 */
router.get('/', async (req, res) => {
  try {
    const { lat, lon, radius = 5000, category } = req.query;

    if (!lat || !lon) {
      return res.status(400).json({
        error: 'Missing required parameters: lat, lon'
      });
    }

    let query = `
      SELECT
        id,
        name,
        category,
        latitude,
        longitude,
        address,
        phone,
        hours,
        verified,
        (
          6371000 * acos(
            cos(radians($1)) * cos(radians(latitude)) *
            cos(radians(longitude) - radians($2)) +
            sin(radians($1)) * sin(radians(latitude))
          )
        ) AS distance
      FROM safe_havens
      WHERE (
        6371000 * acos(
          cos(radians($1)) * cos(radians(latitude)) *
          cos(radians(longitude) - radians($2)) +
          sin(radians($1)) * sin(radians(latitude))
        )
      ) <= $3
    `;

    const params = [parseFloat(lat), parseFloat(lon), parseInt(radius)];

    if (category) {
      query += ` AND category = $4`;
      params.push(category);
    }

    query += ` ORDER BY distance ASC LIMIT 50`;

    const result = await pool.query(query, params);

    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows,
    });
  } catch (error) {
    console.error('Error fetching safe havens:', error);
    res.status(500).json({
      error: 'Failed to fetch safe havens',
      details: error.message
    });
  }
});

/**
 * GET /api/safe-havens/nearest
 * Get the nearest safe haven of any category
 */
router.get('/nearest', async (req, res) => {
  try {
    const { lat, lon } = req.query;

    if (!lat || !lon) {
      return res.status(400).json({
        error: 'Missing required parameters: lat, lon'
      });
    }

    const query = `
      SELECT
        id,
        name,
        category,
        latitude,
        longitude,
        address,
        phone,
        hours,
        verified,
        (
          6371000 * acos(
            cos(radians($1)) * cos(radians(latitude)) *
            cos(radians(longitude) - radians($2)) +
            sin(radians($1)) * sin(radians(latitude))
          )
        ) AS distance
      FROM safe_havens
      ORDER BY distance ASC
      LIMIT 1
    `;

    const result = await pool.query(query, [parseFloat(lat), parseFloat(lon)]);

    if (result.rows.length === 0) {
      return res.status(404).json({
        error: 'No safe havens found'
      });
    }

    res.json({
      success: true,
      data: result.rows[0],
    });
  } catch (error) {
    console.error('Error fetching nearest safe haven:', error);
    res.status(500).json({
      error: 'Failed to fetch nearest safe haven',
      details: error.message
    });
  }
});

module.exports = router;
```

#### Database Migration
`backend/migrations/008_create_safe_havens.sql` (새 파일)

```sql
-- Safe havens table for emergency shelters, embassies, hospitals, etc.
CREATE TABLE IF NOT EXISTS safe_havens (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  category VARCHAR(50) NOT NULL, -- embassy, hospital, un, police, hotel, shelter
  latitude DECIMAL(10, 7) NOT NULL,
  longitude DECIMAL(10, 7) NOT NULL,
  address TEXT,
  phone VARCHAR(50),
  hours TEXT, -- Operating hours (e.g., "24/7" or "Mon-Fri 9-17")
  capacity INTEGER, -- Max people for shelters
  verified BOOLEAN DEFAULT FALSE, -- Verified by admins
  notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for geospatial queries
CREATE INDEX idx_safe_havens_location ON safe_havens(latitude, longitude);
CREATE INDEX idx_safe_havens_category ON safe_havens(category);

-- Sample data for Juba, South Sudan
INSERT INTO safe_havens (name, category, latitude, longitude, address, phone, hours, verified, notes) VALUES
-- Embassies
('미국 대사관 (US Embassy)', 'embassy', 4.8594, 31.5713, 'Kololo Road, Juba', '+211-912-105-188', 'Mon-Fri 8:00-17:00', true, 'Emergency services available 24/7'),
('영국 대사관 (UK Embassy)', 'embassy', 4.8520, 31.5820, 'Thong Ping, EU Compound', '+211-912-105-111', 'Mon-Thu 7:30-15:30', true, 'Consular assistance'),
('케냐 대사관 (Kenya Embassy)', 'embassy', 4.8601, 31.5890, 'Hai Referendum, Juba', '+211-955-061-000', 'Mon-Fri 8:00-16:30', true, null),

-- Hospitals
('Juba Teaching Hospital', 'hospital', 4.8512, 31.5580, 'Airport Road, Juba', '+211-928-888-888', '24/7', true, 'Main public hospital, emergency services'),
('International Hospital Kampala (IHK) Juba', 'hospital', 4.8650, 31.5920, 'Kololo, Juba', '+211-922-000-000', '24/7', true, 'Private hospital, high quality care'),
('Al-Sabah Children''s Hospital', 'hospital', 4.8490, 31.5600, 'Gudele, Juba', '+211-920-000-000', '24/7', true, 'Pediatric emergency care'),

-- UN Facilities
('UNMISS Juba Base', 'un', 4.8780, 31.6010, 'Juba International Airport', '+211-912-177-777', '24/7', true, 'Protection of Civilians site'),
('UNMISS Tomping Base', 'un', 4.8420, 31.5730, 'Tomping, Juba', '+211-912-177-888', '24/7', true, 'POC site, civilian protection'),

-- Police Stations
('Juba Central Police Station', 'police', 4.8530, 31.5800, 'Juba Town Center', '+211-955-000-777', '24/7', true, 'Main police station'),
('Gudele Police Station', 'police', 4.8450, 31.5550, 'Gudele Block 1', '+211-955-000-888', '24/7', true, null),

-- Safe Hotels
('Juba Grand Hotel', 'hotel', 4.8580, 31.5850, 'Hai Referendum, Juba', '+211-928-000-111', '24/7', true, 'Secure compound, 24/7 security'),
('Acacia Village Hotel', 'hotel', 4.8620, 31.5900, 'Kololo Road, Juba', '+211-928-000-222', '24/7', true, 'High security, expat-friendly'),

-- Emergency Shelters
('Red Cross Emergency Shelter', 'shelter', 4.8400, 31.5700, 'Munuki, Juba', '+211-920-111-000', '24/7', true, 'Capacity: 500 people'),
('UN Emergency Shelter - Tomping', 'shelter', 4.8410, 31.5720, 'Tomping, Juba', '+211-912-177-999', '24/7', true, 'Capacity: 1000 people');

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_safe_havens_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER safe_havens_updated_at
BEFORE UPDATE ON safe_havens
FOR EACH ROW
EXECUTE FUNCTION update_safe_havens_updated_at();
```

### 2.2 Frontend Service

`mobile/src/services/api.js`에 추가:

```javascript
/**
 * Safe Havens API
 */
export const safeHavensAPI = {
  /**
   * Get safe havens near a location
   */
  async getNearby(latitude, longitude, radius = 5000, category = null) {
    try {
      let url = `${API_BASE_URL}/safe-havens?lat=${latitude}&lon=${longitude}&radius=${radius}`;

      if (category) {
        url += `&category=${category}`;
      }

      const response = await fetch(url);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch safe havens');
      }

      return data.data;
    } catch (error) {
      console.error('Error fetching safe havens:', error);
      throw error;
    }
  },

  /**
   * Get the nearest safe haven
   */
  async getNearest(latitude, longitude) {
    try {
      const url = `${API_BASE_URL}/safe-havens/nearest?lat=${latitude}&lon=${longitude}`;

      const response = await fetch(url);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch nearest safe haven');
      }

      return data.data;
    } catch (error) {
      console.error('Error fetching nearest safe haven:', error);
      throw error;
    }
  },
};
```

### 2.3 Map Integration

`mobile/src/components/map/SafeHavenMarkers.js` (새 파일)

```javascript
/**
 * Safe haven markers for the map
 */

import React from 'react';
import { Marker } from 'react-native-maps';
import { Colors } from '../../styles';

const CATEGORY_COLORS = {
  embassy: '#4CAF50', // Green - most safe
  hospital: '#2196F3', // Blue - medical
  un: '#00BCD4', // Cyan - international protection
  police: '#FF9800', // Orange - law enforcement
  hotel: '#9C27B0', // Purple - temporary shelter
  shelter: '#795548', // Brown - emergency shelter
};

const CATEGORY_LABELS = {
  embassy: '대사관',
  hospital: '병원',
  un: 'UN',
  police: '경찰',
  hotel: '안전 호텔',
  shelter: '대피소',
};

export default function SafeHavenMarkers({ safeHavens, onMarkerPress }) {
  const getMarkerColor = (category) => {
    return CATEGORY_COLORS[category] || Colors.success;
  };

  return (
    <>
      {safeHavens.map((haven) => (
        <Marker
          key={`safe-haven-${haven.id}`}
          coordinate={{
            latitude: parseFloat(haven.latitude),
            longitude: parseFloat(haven.longitude),
          }}
          pinColor={getMarkerColor(haven.category)}
          title={haven.name}
          description={`${CATEGORY_LABELS[haven.category]} • ${haven.distance ? `${Math.round(haven.distance)}m` : ''}`}
          onPress={() => onMarkerPress && onMarkerPress(haven)}
        />
      ))}
    </>
  );
}
```

### 2.4 Map Screen 통합

`mobile/src/screens/MapScreen.js` 수정:

```javascript
// imports에 추가
import { safeHavensAPI } from '../services/api';
import SafeHavenMarkers from '../components/map/SafeHavenMarkers';

// State 추가
const [safeHavens, setSafeHavens] = useState([]);
const [showSafeHavens, setShowSafeHavens] = useState(true);

// 안전 대피처 로드 함수
const loadSafeHavens = async (latitude, longitude) => {
  try {
    const havens = await safeHavensAPI.getNearby(latitude, longitude, 10000); // 10km radius
    setSafeHavens(havens);
  } catch (error) {
    console.error('Failed to load safe havens:', error);
  }
};

// useEffect에서 호출
useEffect(() => {
  if (location) {
    loadSafeHavens(location.latitude, location.longitude);
  }
}, [location]);

// MapView 내부에 추가
{showSafeHavens && (
  <SafeHavenMarkers
    safeHavens={safeHavens}
    onMarkerPress={(haven) => {
      // Show safe haven details in a bottom sheet or modal
      console.log('Safe haven selected:', haven);
    }}
  />
)}
```

---

## 3. 긴급 SOS 버튼

### 3.1 Backend API

`backend/routes/emergency.js` 수정/추가:

```javascript
/**
 * POST /api/emergency/sos
 * Trigger emergency SOS alert
 */
router.post('/sos', async (req, res) => {
  try {
    const {
      userId,
      latitude,
      longitude,
      message,
    } = req.body;

    if (!userId || !latitude || !longitude) {
      return res.status(400).json({
        error: 'Missing required fields: userId, latitude, longitude'
      });
    }

    // 1. Save SOS event to database
    const sosQuery = `
      INSERT INTO sos_events (user_id, latitude, longitude, message, status)
      VALUES ($1, $2, $3, $4, 'active')
      RETURNING id, created_at
    `;

    const sosResult = await pool.query(sosQuery, [
      userId,
      latitude,
      longitude,
      message || 'Emergency SOS activated',
    ]);

    const sosId = sosResult.rows[0].id;

    // 2. Get user's emergency contacts
    // Note: In production, emergency contacts would be stored in the database
    // For now, we'll return the SOS ID and let the mobile app handle notifications

    // 3. Get nearest safe haven
    const safeHavenQuery = `
      SELECT
        id,
        name,
        category,
        latitude,
        longitude,
        address,
        phone,
        (
          6371000 * acos(
            cos(radians($1)) * cos(radians(latitude)) *
            cos(radians(longitude) - radians($2)) +
            sin(radians($1)) * sin(radians(latitude))
          )
        ) AS distance
      FROM safe_havens
      ORDER BY distance ASC
      LIMIT 1
    `;

    const safeHavenResult = await pool.query(safeHavenQuery, [latitude, longitude]);
    const nearestSafeHaven = safeHavenResult.rows[0] || null;

    // 4. Log the event
    console.log(`[SOS] User ${userId} activated SOS at (${latitude}, ${longitude})`);

    res.json({
      success: true,
      sosId,
      timestamp: sosResult.rows[0].created_at,
      nearestSafeHaven,
      message: 'SOS alert sent successfully',
    });

  } catch (error) {
    console.error('Error processing SOS:', error);
    res.status(500).json({
      error: 'Failed to process SOS',
      details: error.message,
    });
  }
});

/**
 * POST /api/emergency/sos/:sosId/cancel
 * Cancel an active SOS
 */
router.post('/sos/:sosId/cancel', async (req, res) => {
  try {
    const { sosId } = req.params;
    const { userId } = req.body;

    const query = `
      UPDATE sos_events
      SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
      WHERE id = $1 AND user_id = $2 AND status = 'active'
      RETURNING id
    `;

    const result = await pool.query(query, [sosId, userId]);

    if (result.rows.length === 0) {
      return res.status(404).json({
        error: 'SOS not found or already resolved',
      });
    }

    console.log(`[SOS] User ${userId} cancelled SOS ${sosId}`);

    res.json({
      success: true,
      message: 'SOS cancelled successfully',
    });

  } catch (error) {
    console.error('Error cancelling SOS:', error);
    res.status(500).json({
      error: 'Failed to cancel SOS',
      details: error.message,
    });
  }
});
```

### 3.2 Database Migration

`backend/migrations/009_create_sos_events.sql` (새 파일)

```sql
-- SOS events table
CREATE TABLE IF NOT EXISTS sos_events (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  latitude DECIMAL(10, 7) NOT NULL,
  longitude DECIMAL(10, 7) NOT NULL,
  message TEXT,
  status VARCHAR(20) DEFAULT 'active', -- active, resolved, cancelled
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  resolved_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_sos_events_user_id ON sos_events(user_id);
CREATE INDEX idx_sos_events_status ON sos_events(status);
CREATE INDEX idx_sos_events_created_at ON sos_events(created_at DESC);

-- Trigger for updated_at
CREATE TRIGGER sos_events_updated_at
BEFORE UPDATE ON sos_events
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();
```

### 3.3 Enhanced EmergencyButton Component

`mobile/src/components/EmergencyButton.js` 수정:

```javascript
/**
 * Enhanced Emergency SOS Button
 * - 3초 길게 눌러서 활성화
 * - 진동 및 소리 피드백
 * - 긴급 연락망에 자동 알림
 * - 가장 가까운 안전 대피처 표시
 */

import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Alert,
  Vibration,
  Platform,
} from 'react-native';
import * as Location from 'expo-location';
import * as SMS from 'expo-sms';
import { Colors, Typography, Spacing } from '../styles';
import { emergencyContactsStorage } from '../services/storage';
import { emergencyAPI, safeHavensAPI } from '../services/api';
import Icon from './icons/Icon';

const LONG_PRESS_DURATION = 3000; // 3 seconds

export default function EmergencyButton({ onSOSActivated }) {
  const [pressing, setPressing] = useState(false);
  const [active, setActive] = useState(false);
  const progressAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const longPressTimer = useRef(null);

  const startPress = () => {
    setPressing(true);

    // Haptic feedback
    Vibration.vibrate([0, 100]);

    // Progress animation
    Animated.timing(progressAnim, {
      toValue: 1,
      duration: LONG_PRESS_DURATION,
      useNativeDriver: false,
    }).start();

    // Scale animation
    Animated.spring(scaleAnim, {
      toValue: 1.1,
      useNativeDriver: true,
    }).start();

    // Long press timer
    longPressTimer.current = setTimeout(() => {
      activateSOS();
    }, LONG_PRESS_DURATION);
  };

  const endPress = () => {
    setPressing(false);

    if (longPressTimer.current) {
      clearTimeout(longPressTimer.current);
    }

    // Reset animations
    Animated.parallel([
      Animated.timing(progressAnim, {
        toValue: 0,
        duration: 200,
        useNativeDriver: false,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const activateSOS = async () => {
    try {
      // Strong vibration pattern
      Vibration.vibrate([0, 500, 200, 500, 200, 500]);

      setActive(true);
      setPressing(false);

      // Get current location
      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });

      const { latitude, longitude } = location.coords;

      // Send SOS to backend
      const sosResponse = await emergencyAPI.triggerSOS({
        userId: 1, // TODO: Get from auth context
        latitude,
        longitude,
        message: 'Emergency SOS activated from VeriSafe app',
      });

      // Get emergency contacts
      const contacts = await emergencyContactsStorage.getAll();

      // Prepare SOS message
      const sosMessage = `[VeriSafe 긴급 알림]\n\n긴급 SOS가 발동되었습니다!\n\n위치: https://maps.google.com/?q=${latitude},${longitude}\n\n${sosResponse.nearestSafeHaven ? `가장 가까운 안전 대피처:\n${sosResponse.nearestSafeHaven.name}\n거리: ${Math.round(sosResponse.nearestSafeHaven.distance)}m\n연락처: ${sosResponse.nearestSafeHaven.phone || 'N/A'}` : ''}`;

      // Send SMS to emergency contacts
      if (contacts.length > 0) {
        const phoneNumbers = contacts.map(c => c.phone);

        const isAvailable = await SMS.isAvailableAsync();
        if (isAvailable) {
          await SMS.sendSMSAsync(phoneNumbers, sosMessage);
        } else {
          console.warn('SMS not available on this device');
        }
      }

      // Show confirmation
      Alert.alert(
        'SOS 발동됨',
        `긴급 연락망 ${contacts.length}명에게 알림이 전송되었습니다.\n\n${sosResponse.nearestSafeHaven ? `가장 가까운 안전 대피처:\n${sosResponse.nearestSafeHaven.name}\n거리: ${Math.round(sosResponse.nearestSafeHaven.distance)}m` : ''}`,
        [
          {
            text: 'SOS 취소',
            style: 'destructive',
            onPress: () => cancelSOS(sosResponse.sosId),
          },
          {
            text: '확인',
            style: 'default',
          },
        ]
      );

      // Callback to parent component
      if (onSOSActivated) {
        onSOSActivated(sosResponse);
      }

    } catch (error) {
      console.error('Failed to activate SOS:', error);
      Alert.alert(
        '오류',
        'SOS 발동에 실패했습니다. 직접 긴급 연락처에 전화하세요.',
        [
          { text: '확인', style: 'default' },
        ]
      );
    }
  };

  const cancelSOS = async (sosId) => {
    try {
      await emergencyAPI.cancelSOS(sosId, 1); // TODO: Get userId from auth context
      setActive(false);
      Alert.alert('확인', 'SOS가 취소되었습니다.');
    } catch (error) {
      console.error('Failed to cancel SOS:', error);
    }
  };

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <View style={styles.container}>
      <Animated.View style={[styles.buttonContainer, { transform: [{ scale: scaleAnim }] }]}>
        <TouchableOpacity
          activeOpacity={0.9}
          onPressIn={startPress}
          onPressOut={endPress}
          style={[
            styles.button,
            active && styles.buttonActive,
          ]}
        >
          <View style={styles.buttonContent}>
            <Icon
              name="warning"
              size={32}
              color={Colors.textInverse}
            />
            <Text style={styles.buttonText}>
              {pressing ? '계속 누르세요...' : active ? 'SOS 발동됨' : 'SOS'}
            </Text>
            {!pressing && !active && (
              <Text style={styles.buttonHint}>3초간 길게 누르기</Text>
            )}
          </View>

          {pressing && (
            <Animated.View style={[styles.progressBar, { width: progressWidth }]} />
          )}
        </TouchableOpacity>
      </Animated.View>

      {active && (
        <View style={styles.activeIndicator}>
          <View style={styles.pulseDot} />
          <Text style={styles.activeText}>긴급 모드</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
  },
  buttonContainer: {
    position: 'relative',
  },
  button: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: Colors.danger,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.danger,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
    borderWidth: 4,
    borderColor: Colors.textInverse,
    overflow: 'hidden',
  },
  buttonActive: {
    backgroundColor: '#D32F2F',
    shadowOpacity: 0.6,
  },
  buttonContent: {
    alignItems: 'center',
    gap: 4,
    zIndex: 2,
  },
  buttonText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: '700',
    fontSize: 16,
  },
  buttonHint: {
    ...Typography.captionSmall,
    color: Colors.textInverse,
    opacity: 0.8,
    fontSize: 10,
  },
  progressBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    height: '100%',
    backgroundColor: '#B71C1C',
    zIndex: 1,
  },
  activeIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: Spacing.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    backgroundColor: Colors.danger + '20',
    borderRadius: 20,
  },
  pulseDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: Colors.danger,
  },
  activeText: {
    ...Typography.labelSmall,
    color: Colors.danger,
    fontWeight: '700',
  },
});
```

### 3.4 API Service 추가

`mobile/src/services/api.js`에 추가:

```javascript
/**
 * Emergency API
 */
export const emergencyAPI = {
  /**
   * Trigger emergency SOS
   */
  async triggerSOS(sosData) {
    try {
      const response = await fetch(`${API_BASE_URL}/emergency/sos`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sosData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to trigger SOS');
      }

      return data;
    } catch (error) {
      console.error('Error triggering SOS:', error);
      throw error;
    }
  },

  /**
   * Cancel SOS
   */
  async cancelSOS(sosId, userId) {
    try {
      const response = await fetch(`${API_BASE_URL}/emergency/sos/${sosId}/cancel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ userId }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to cancel SOS');
      }

      return data;
    } catch (error) {
      console.error('Error cancelling SOS:', error);
      throw error;
    }
  },
};
```

---

## 4. 위험 지역 진입 경고

### 4.1 Geofencing Service

`mobile/src/services/geofencing.js` (새 파일)

```javascript
/**
 * Geofencing service for danger zone monitoring
 */

import * as TaskManager from 'expo-task-manager';
import * as Location from 'expo-location';
import { Alert, Vibration } from 'react-native';
import { riskAssessmentAPI } from './api';

const GEOFENCE_TASK_NAME = 'DANGER_ZONE_GEOFENCE';
const DANGER_THRESHOLD = 70; // Risk score threshold
const DANGER_RADIUS = 1000; // 1km

class GeofencingService {
  constructor() {
    this.isMonitoring = false;
    this.currentLocation = null;
    this.dangerZones = [];
    this.lastAlertTime = {};
  }

  /**
   * Start monitoring for danger zones
   */
  async startMonitoring() {
    try {
      // Request background location permission
      const { status } = await Location.requestBackgroundPermissionsAsync();

      if (status !== 'granted') {
        Alert.alert(
          '권한 필요',
          '위험 지역 경고를 받으려면 백그라운드 위치 권한이 필요합니다.'
        );
        return false;
      }

      // Define the task
      TaskManager.defineTask(GEOFENCE_TASK_NAME, async ({ data, error }) => {
        if (error) {
          console.error('Geofence task error:', error);
          return;
        }

        if (data) {
          const { locations } = data;
          const location = locations[0];

          if (location) {
            await this.checkDangerZones(
              location.coords.latitude,
              location.coords.longitude
            );
          }
        }
      });

      // Start location updates
      await Location.startLocationUpdatesAsync(GEOFENCE_TASK_NAME, {
        accuracy: Location.Accuracy.Balanced,
        timeInterval: 30000, // 30 seconds
        distanceInterval: 100, // 100 meters
        showsBackgroundLocationIndicator: true,
        foregroundService: {
          notificationTitle: 'VeriSafe 보호 활성화',
          notificationBody: '위험 지역을 모니터링 중입니다',
          notificationColor: '#FF5252',
        },
      });

      this.isMonitoring = true;
      console.log('Danger zone monitoring started');
      return true;

    } catch (error) {
      console.error('Failed to start monitoring:', error);
      return false;
    }
  }

  /**
   * Stop monitoring
   */
  async stopMonitoring() {
    try {
      await Location.stopLocationUpdatesAsync(GEOFENCE_TASK_NAME);
      this.isMonitoring = false;
      console.log('Danger zone monitoring stopped');
      return true;
    } catch (error) {
      console.error('Failed to stop monitoring:', error);
      return false;
    }
  }

  /**
   * Check if current location is in a danger zone
   */
  async checkDangerZones(latitude, longitude) {
    try {
      // Get risk assessment for current location
      const assessment = await riskAssessmentAPI.getAssessment(latitude, longitude);

      if (assessment.riskScore >= DANGER_THRESHOLD) {
        // Check if we recently alerted for this zone
        const zoneKey = `${Math.round(latitude * 100)}_${Math.round(longitude * 100)}`;
        const now = Date.now();
        const lastAlert = this.lastAlertTime[zoneKey] || 0;

        // Only alert once every 5 minutes for the same zone
        if (now - lastAlert > 5 * 60 * 1000) {
          this.triggerDangerAlert(assessment);
          this.lastAlertTime[zoneKey] = now;
        }
      }

    } catch (error) {
      console.error('Failed to check danger zones:', error);
    }
  }

  /**
   * Trigger danger zone alert
   */
  triggerDangerAlert(assessment) {
    // Strong vibration pattern
    Vibration.vibrate([0, 500, 200, 500, 200, 500]);

    // Show full-screen alert
    Alert.alert(
      '⚠️ 위험 지역 진입 경고',
      `현재 위치의 위험도가 높습니다!\n\n위험도: ${assessment.riskScore}/100\n주요 위험: ${assessment.primaryThreat || '알 수 없음'}\n\n안전한 경로로 우회하거나 즉시 대피하세요.`,
      [
        {
          text: '안전 대피처 찾기',
          onPress: () => {
            // Navigate to safe haven
            // This would be handled by the navigation system
          },
        },
        {
          text: '경로 재탐색',
          onPress: () => {
            // Trigger route recalculation avoiding this area
          },
        },
        {
          text: '확인',
          style: 'cancel',
        },
      ],
      {
        cancelable: false, // Force user to acknowledge
      }
    );

    // TODO: Voice warning using Text-to-Speech
    // speak('위험 지역입니다. 즉시 대피하세요.');
  }

  /**
   * Get monitoring status
   */
  getStatus() {
    return {
      isMonitoring: this.isMonitoring,
      dangerZonesCount: this.dangerZones.length,
    };
  }
}

export default new GeofencingService();
```

### 4.2 Settings Integration

`mobile/src/screens/SettingsScreen.js` 수정:

```javascript
// imports에 추가
import geofencingService from '../services/geofencing';

// State 추가
const [dangerZoneAlerts, setDangerZoneAlerts] = useState(true);

// Load settings
useEffect(() => {
  loadSettings();
}, []);

const loadSettings = async () => {
  try {
    const stored = await AsyncStorage.getItem('@verisafe_settings');
    if (stored) {
      const settings = JSON.parse(stored);
      setDangerZoneAlerts(settings.dangerZoneAlerts ?? true);
    }
  } catch (error) {
    console.error('Failed to load settings:', error);
  }
};

// Toggle danger zone alerts
const toggleDangerZoneAlerts = async (value) => {
  try {
    setDangerZoneAlerts(value);

    // Save to storage
    const stored = await AsyncStorage.getItem('@verisafe_settings');
    const settings = stored ? JSON.parse(stored) : {};
    settings.dangerZoneAlerts = value;
    await AsyncStorage.setItem('@verisafe_settings', JSON.stringify(settings));

    // Start/stop monitoring
    if (value) {
      const started = await geofencingService.startMonitoring();
      if (!started) {
        // Revert if failed
        setDangerZoneAlerts(false);
      }
    } else {
      await geofencingService.stopMonitoring();
    }

  } catch (error) {
    console.error('Failed to toggle danger zone alerts:', error);
    Alert.alert('오류', '설정 변경에 실패했습니다.');
  }
};

// Add to Settings UI
<View style={styles.settingItem}>
  <View style={styles.settingLeft}>
    <Icon name="warning" size={24} color={Colors.danger} />
    <View>
      <Text style={styles.settingLabel}>위험 지역 진입 경고</Text>
      <Text style={styles.settingDescription}>
        위험 지역에 진입하면 즉시 알림
      </Text>
    </View>
  </View>
  <Switch
    value={dangerZoneAlerts}
    onValueChange={toggleDangerZoneAlerts}
    trackColor={{ false: Colors.border, true: Colors.danger + '60' }}
    thumbColor={dangerZoneAlerts ? Colors.danger : Colors.textTertiary}
  />
</View>
```

---

## 5. 안전 체크인 시스템

### 5.1 Backend API

`backend/routes/safety-checkin.js` (새 파일)

```javascript
const express = require('express');
const router = express.Router();
const pool = require('../db');

/**
 * POST /api/safety-checkin/register
 * Register a safety check-in for a route
 */
router.post('/register', async (req, res) => {
  try {
    const {
      userId,
      routeId,
      estimatedArrivalTime,
      destinationLat,
      destinationLon,
    } = req.body;

    if (!userId || !estimatedArrivalTime) {
      return res.status(400).json({
        error: 'Missing required fields: userId, estimatedArrivalTime'
      });
    }

    const query = `
      INSERT INTO safety_checkins (
        user_id,
        route_id,
        estimated_arrival_time,
        destination_lat,
        destination_lon,
        status
      ) VALUES ($1, $2, $3, $4, $5, 'active')
      RETURNING id, created_at
    `;

    const result = await pool.query(query, [
      userId,
      routeId || null,
      estimatedArrivalTime,
      destinationLat || null,
      destinationLon || null,
    ]);

    console.log(`[Safety Check-in] Registered for user ${userId}, ETA: ${estimatedArrivalTime}`);

    res.json({
      success: true,
      checkinId: result.rows[0].id,
      createdAt: result.rows[0].created_at,
      message: 'Safety check-in registered',
    });

  } catch (error) {
    console.error('Error registering safety check-in:', error);
    res.status(500).json({
      error: 'Failed to register safety check-in',
      details: error.message,
    });
  }
});

/**
 * POST /api/safety-checkin/:checkinId/confirm
 * Confirm safe arrival
 */
router.post('/:checkinId/confirm', async (req, res) => {
  try {
    const { checkinId } = req.params;
    const { userId } = req.body;

    const query = `
      UPDATE safety_checkins
      SET
        status = 'confirmed',
        confirmed_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
      WHERE id = $1 AND user_id = $2 AND status = 'active'
      RETURNING id
    `;

    const result = await pool.query(query, [checkinId, userId]);

    if (result.rows.length === 0) {
      return res.status(404).json({
        error: 'Check-in not found or already confirmed',
      });
    }

    console.log(`[Safety Check-in] User ${userId} confirmed arrival (check-in ${checkinId})`);

    res.json({
      success: true,
      message: 'Arrival confirmed',
    });

  } catch (error) {
    console.error('Error confirming check-in:', error);
    res.status(500).json({
      error: 'Failed to confirm check-in',
      details: error.message,
    });
  }
});

/**
 * GET /api/safety-checkin/overdue
 * Get all overdue check-ins that need alerts
 */
router.get('/overdue', async (req, res) => {
  try {
    const query = `
      SELECT
        id,
        user_id,
        route_id,
        estimated_arrival_time,
        destination_lat,
        destination_lon,
        created_at,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - estimated_arrival_time)) AS overdue_seconds
      FROM safety_checkins
      WHERE status = 'active'
        AND estimated_arrival_time < CURRENT_TIMESTAMP - INTERVAL '30 minutes'
      ORDER BY estimated_arrival_time ASC
    `;

    const result = await pool.query(query);

    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows,
    });

  } catch (error) {
    console.error('Error fetching overdue check-ins:', error);
    res.status(500).json({
      error: 'Failed to fetch overdue check-ins',
      details: error.message,
    });
  }
});

module.exports = router;
```

### 5.2 Database Migration

`backend/migrations/010_create_safety_checkins.sql` (새 파일)

```sql
-- Safety check-ins table
CREATE TABLE IF NOT EXISTS safety_checkins (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  route_id INTEGER,
  estimated_arrival_time TIMESTAMP NOT NULL,
  destination_lat DECIMAL(10, 7),
  destination_lon DECIMAL(10, 7),
  status VARCHAR(20) DEFAULT 'active', -- active, confirmed, missed, cancelled
  confirmed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_safety_checkins_user_id ON safety_checkins(user_id);
CREATE INDEX idx_safety_checkins_status ON safety_checkins(status);
CREATE INDEX idx_safety_checkins_eta ON safety_checkins(estimated_arrival_time);

-- Trigger for updated_at
CREATE TRIGGER safety_checkins_updated_at
BEFORE UPDATE ON safety_checkins
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();
```

### 5.3 Frontend Service

`mobile/src/services/safetyCheckin.js` (새 파일)

```javascript
/**
 * Safety check-in service
 * Automatically registers check-ins when navigation starts
 * Sends alerts if user doesn't confirm arrival
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Notifications from 'expo-notifications';
import { Alert } from 'react-native';
import { emergencyContactsStorage } from './storage';

const STORAGE_KEY = '@verisafe_active_checkin';

class SafetyCheckinService {
  constructor() {
    this.activeCheckin = null;
    this.checkInterval = null;
  }

  /**
   * Register a safety check-in
   */
  async register(route, estimatedArrivalTime) {
    try {
      const checkin = {
        id: `checkin_${Date.now()}`,
        routeId: route.id || null,
        origin: route.origin,
        destination: route.destination,
        estimatedArrivalTime,
        status: 'active',
        createdAt: new Date().toISOString(),
      };

      // Save to local storage
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(checkin));
      this.activeCheckin = checkin;

      // Schedule notification for ETA + 30 minutes
      const etaPlus30 = new Date(estimatedArrivalTime);
      etaPlus30.setMinutes(etaPlus30.getMinutes() + 30);

      await Notifications.scheduleNotificationAsync({
        content: {
          title: '⚠️ 안전 체크인 필요',
          body: '목적지에 도착하셨나요? 안전을 확인해주세요.',
          sound: true,
          priority: Notifications.AndroidNotificationPriority.HIGH,
        },
        trigger: {
          date: etaPlus30,
        },
      });

      // Start monitoring
      this.startMonitoring();

      console.log('[Safety Check-in] Registered:', checkin);
      return checkin;

    } catch (error) {
      console.error('Failed to register safety check-in:', error);
      return null;
    }
  }

  /**
   * Confirm safe arrival
   */
  async confirm() {
    try {
      if (!this.activeCheckin) {
        return false;
      }

      // Update status
      this.activeCheckin.status = 'confirmed';
      this.activeCheckin.confirmedAt = new Date().toISOString();

      // Clear from storage
      await AsyncStorage.removeItem(STORAGE_KEY);

      // Cancel scheduled notifications
      await Notifications.cancelAllScheduledNotificationsAsync();

      // Stop monitoring
      this.stopMonitoring();

      console.log('[Safety Check-in] Confirmed');
      this.activeCheckin = null;
      return true;

    } catch (error) {
      console.error('Failed to confirm check-in:', error);
      return false;
    }
  }

  /**
   * Cancel check-in
   */
  async cancel() {
    try {
      if (!this.activeCheckin) {
        return false;
      }

      this.activeCheckin.status = 'cancelled';
      await AsyncStorage.removeItem(STORAGE_KEY);
      await Notifications.cancelAllScheduledNotificationsAsync();
      this.stopMonitoring();

      console.log('[Safety Check-in] Cancelled');
      this.activeCheckin = null;
      return true;

    } catch (error) {
      console.error('Failed to cancel check-in:', error);
      return false;
    }
  }

  /**
   * Start monitoring for overdue check-ins
   */
  startMonitoring() {
    if (this.checkInterval) {
      return;
    }

    // Check every minute
    this.checkInterval = setInterval(() => {
      this.checkOverdue();
    }, 60 * 1000);
  }

  /**
   * Stop monitoring
   */
  stopMonitoring() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }

  /**
   * Check if check-in is overdue
   */
  async checkOverdue() {
    try {
      if (!this.activeCheckin || this.activeCheckin.status !== 'active') {
        return;
      }

      const eta = new Date(this.activeCheckin.estimatedArrivalTime);
      const now = new Date();
      const etaPlus30 = new Date(eta);
      etaPlus30.setMinutes(etaPlus30.getMinutes() + 30);

      // If current time is past ETA + 30 minutes
      if (now > etaPlus30) {
        await this.triggerMissedAlert();
      }

    } catch (error) {
      console.error('Error checking overdue:', error);
    }
  }

  /**
   * Trigger alert for missed check-in
   */
  async triggerMissedAlert() {
    try {
      // Update status
      this.activeCheckin.status = 'missed';
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(this.activeCheckin));

      // Get emergency contacts
      const contacts = await emergencyContactsStorage.getAll();

      // Send alert to emergency contacts
      const message = `[VeriSafe 안전 체크인 경고]\n\n사용자가 예정된 도착 시간에 안전 체크인을 하지 않았습니다.\n\n목적지: ${this.activeCheckin.destination.name || '위치 정보 없음'}\n예상 도착: ${new Date(this.activeCheckin.estimatedArrivalTime).toLocaleString('ko-KR')}\n\n연락하여 안전을 확인해주세요.`;

      // TODO: Send SMS/Push notifications to contacts
      console.log('[Safety Check-in] MISSED - Alerting contacts:', message);

      // Show alert to user
      Alert.alert(
        '⚠️ 안전 체크인 놓침',
        '예정된 도착 시간이 지났습니다. 긴급 연락망에 알림이 전송되었습니다.',
        [
          {
            text: '안전 확인',
            onPress: () => this.confirm(),
          },
          {
            text: '나중에',
            style: 'cancel',
          },
        ]
      );

    } catch (error) {
      console.error('Failed to trigger missed alert:', error);
    }
  }

  /**
   * Get active check-in
   */
  async getActive() {
    try {
      if (this.activeCheckin) {
        return this.activeCheckin;
      }

      const stored = await AsyncStorage.getItem(STORAGE_KEY);
      if (stored) {
        this.activeCheckin = JSON.parse(stored);
        this.startMonitoring();
        return this.activeCheckin;
      }

      return null;

    } catch (error) {
      console.error('Failed to get active check-in:', error);
      return null;
    }
  }
}

export default new SafetyCheckinService();
```

### 5.4 Route Planning Integration

`mobile/src/screens/RoutePlanningScreen.js` 수정:

```javascript
// imports에 추가
import safetyCheckinService from '../services/safetyCheckin';

// 경로 계산 성공 후 체크인 등록
const handleRouteCalculated = async (route) => {
  try {
    // Set route
    setRoute(route);

    // Register safety check-in
    if (route.duration) {
      const eta = new Date();
      eta.setSeconds(eta.getSeconds() + route.duration);

      const checkin = await safetyCheckinService.register(route, eta);

      if (checkin) {
        Alert.alert(
          '안전 체크인 등록됨',
          `목적지 도착 예정: ${eta.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}\n\n도착 후 안전 체크인을 해주세요.`,
          [{ text: '확인' }]
        );
      }
    }

  } catch (error) {
    console.error('Route calculation error:', error);
  }
};

// 목적지 도착 시 체크인 확인
const handleArrival = () => {
  Alert.alert(
    '목적지 도착',
    '목적지에 안전하게 도착하셨나요?',
    [
      {
        text: '예, 안전합니다',
        onPress: async () => {
          await safetyCheckinService.confirm();
          Alert.alert('확인', '안전 체크인이 완료되었습니다.');
        },
      },
      {
        text: '아니오',
        style: 'cancel',
      },
    ]
  );
};
```

---

## 테스트 계획

### Day 13-14: 통합 테스트

#### 1. 긴급 연락망 테스트
- [ ] 연락처 추가/수정/삭제
- [ ] 최대 5명 제한 확인
- [ ] 우선순위 정렬
- [ ] 전화 걸기 기능

#### 2. 안전 대피처 테스트
- [ ] 지도에 마커 표시
- [ ] 카테고리별 필터링
- [ ] 거리 계산 정확도
- [ ] 가장 가까운 대피처 찾기

#### 3. SOS 버튼 테스트
- [ ] 3초 길게 누르기 동작
- [ ] 진동 및 시각 피드백
- [ ] SMS 전송 (실제 전송 테스트)
- [ ] 가장 가까운 대피처 표시
- [ ] SOS 취소 기능

#### 4. 위험 지역 경고 테스트
- [ ] 백그라운드 위치 권한
- [ ] Geofencing 동작
- [ ] 알림 표시 (진동, 소리, Alert)
- [ ] 중복 알림 방지 (5분 간격)

#### 5. 안전 체크인 테스트
- [ ] 경로 시작 시 자동 등록
- [ ] ETA 계산
- [ ] ETA+30분 알림
- [ ] 체크인 확인 동작
- [ ] 긴급 연락망 자동 알림

---

## 성공 지표

### 기술적 지표
1. **SOS 응답 시간**: < 3초
2. **위험 지역 감지 정확도**: > 95%
3. **안전 대피처 검색 속도**: < 1초
4. **백그라운드 배터리 소모**: < 5%/hour

### 사용성 지표
1. **긴급 연락망 등록률**: > 80% of users
2. **SOS 오발동률**: < 5%
3. **안전 체크인 완료율**: > 90%
4. **사용자 만족도**: > 4.5/5.0

---

## 다음 단계 (Phase 2)

Phase 1 완료 후:
1. ETA 표시 기능
2. 음성 위험 경고
3. 경로 공유 기능
4. 야간 모드 (밤에 더 안전한 경로)
5. 실시간 위험 푸시 알림

---

## 참고 자료

- React Native Geolocation: https://docs.expo.dev/versions/latest/sdk/location/
- Background Tasks: https://docs.expo.dev/versions/latest/sdk/task-manager/
- Local Notifications: https://docs.expo.dev/versions/latest/sdk/notifications/
- SMS: https://docs.expo.dev/versions/latest/sdk/sms/
- Haptics: https://docs.expo.dev/versions/latest/sdk/haptics/
