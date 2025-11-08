/**
 * Icon - 통합 아이콘 컴포넌트
 * MaterialIcons를 래핑하여 일관된 아이콘 사용
 * 
 * 사용법:
 * <Icon name="route" size={24} color={Colors.primary} />
 */

import React from 'react';
import { MaterialIcons } from '@expo/vector-icons';

/**
 * 아이콘 이름 매핑 (이모지 → MaterialIcons 이름)
 */
export const ICON_MAP = {
  // Navigation
  route: 'route',
  location: 'location-on',
  locationOn: 'location-on',  // 별칭
  locationOff: 'location-off',
  search: 'search',
  map: 'map',
  navigation: 'navigation',
  myLocation: 'my-location',
  target: 'my-location',  // 목적지 아이콘 (navigation 대신)
  layers: 'layers',  // 레이어 버튼
  
  // Hazard
  warning: 'warning',
  checkpoint: 'security',
  protest: 'groups',
  conflict: 'dangerous',
  roadDamage: 'construction',
  naturalDisaster: 'flash-on',
  safeHaven: 'shield',  // 안전 대피처
  other: 'help-outline',
  
  // Transportation
  car: 'directions-car',
  walking: 'directions-walk',
  bicycle: 'directions-bike',
  
  // Action
  save: 'bookmark',
  share: 'share',
  report: 'report',
  close: 'close',
  camera: 'camera-alt',
  'add-photo': 'add-a-photo',
  add: 'add',
  edit: 'edit',
  delete: 'delete',
  deleteForever: 'delete-forever',
  mic: 'mic',
  stop: 'stop',
  send: 'send',
  bookmark: 'bookmark',
  bookmarkBorder: 'bookmark-border',
  history: 'history',
  directions: 'directions',
  phone: 'phone',

  // Checkboxes
  'check-box': 'check-box',
  'check-box-outline-blank': 'check-box-outline-blank',

  // Navigation arrows
  'chevron-right': 'chevron-right',
  'chevron-left': 'chevron-left',
  'chevron-down': 'keyboard-arrow-down',
  'arrow-forward': 'arrow-forward',
  chevronRight: 'chevron-right',  // camelCase alias
  chevronLeft: 'chevron-left',    // camelCase alias
  chevronDown: 'keyboard-arrow-down',  // camelCase alias

  // Other
  article: 'article',
  person: 'person',
  people: 'people',
  work: 'work',
  contactPhone: 'contact-phone',
  database: 'storage',  // 데이터베이스 아이콘
  refresh: 'refresh',   // 새로고침 아이콘
  info: 'info',         // 정보 아이콘
  settings: 'settings',
  notifications: 'notifications',
  language: 'language',
  sync: 'sync',
  download: 'file-download',

  // Time & Distance
  time: 'access-time',
  distance: 'place',
  
  // Route Types
  safe: 'shield',
  fast: 'flash-on',
  alternative: 'alt-route',
  
  // Landmarks
  flight: 'flight',
  'local-hospital': 'local-hospital',
  'account-balance': 'account-balance',
  hotel: 'hotel',
};

/**
 * 이모지 타입을 아이콘 이름으로 변환하는 헬퍼 함수
 */
export const getIconName = (emojiType) => {
  const emojiMap = {
    '🗺️': 'route',
    '📍': 'location',
    '🔍': 'search',
    '⚠️': 'warning',
    '🚗': 'car',
    '🚶': 'walking',
    '🚴': 'bicycle',
    '⭐': 'save',
    '📤': 'share',
    '⏱️': 'time',
    '🛡️': 'safe',
    '⚡': 'fast',
    '🏥': 'local-hospital',
    '🏛️': 'account-balance',
    '🏨': 'hotel',
    '✈️': 'flight',
    '👥': 'groups',
    '🚧': 'construction',
    '💥': 'flash-on',
    '❓': 'help-outline',
  };
  
  return emojiMap[emojiType] || 'help-outline';
};

/**
 * Icon 컴포넌트
 */
export default function Icon({ 
  name, 
  size = 24, 
  color = '#000000', 
  ...props 
}) {
  // name이 ICON_MAP에 있으면 사용, 없으면 그대로 사용
  const iconName = ICON_MAP[name] || name;
  
  return (
    <MaterialIcons 
      name={iconName} 
      size={size} 
      color={color} 
      {...props} 
    />
  );
}

