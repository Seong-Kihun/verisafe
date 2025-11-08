/**
 * 로컬 저장소 서비스 (AsyncStorage)
 * 사용자 프로필, 즐겨찾기, 경로 기록 등을 저장
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

// 저장소 키
const KEYS = {
  USER_PROFILE: '@verisafe:user_profile',
  SAVED_PLACES: '@verisafe:saved_places',
  RECENT_ROUTES: '@verisafe:recent_routes',
  MY_REPORTS: '@verisafe:my_reports',
  EMERGENCY_CONTACTS: '@verisafe:emergency_contacts',
  SETTINGS: '@verisafe:settings',
  STATS: '@verisafe:stats',
  ONBOARDING_COMPLETED: '@verisafe:onboarding_completed',
  USER_COUNTRY: '@verisafe:user_country',
};

// 사용자 프로필
export const userProfileStorage = {
  get: async () => {
    try {
      const data = await AsyncStorage.getItem(KEYS.USER_PROFILE);
      return data ? JSON.parse(data) : {
        id: 1, // 기본 사용자 ID
        name: 'KOICA Worker',
        email: 'worker@koica.go.kr',
        phone: '+211-XXX-XXXX',
        organization: 'KOICA',
        role: 'user',
      };
    } catch (error) {
      console.error('Failed to load user profile:', error);
      // 에러 발생 시에도 기본 프로필 반환
      return {
        id: 1,
        name: 'KOICA Worker',
        email: 'worker@koica.go.kr',
        phone: '+211-XXX-XXXX',
        organization: 'KOICA',
        role: 'user',
      };
    }
  },

  save: async (profile) => {
    try {
      await AsyncStorage.setItem(KEYS.USER_PROFILE, JSON.stringify(profile));
      return true;
    } catch (error) {
      console.error('Failed to save user profile:', error);
      return false;
    }
  },
};

// 즐겨찾기 장소
export const savedPlacesStorage = {
  getAll: async () => {
    try {
      const data = await AsyncStorage.getItem(KEYS.SAVED_PLACES);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Failed to load saved places:', error);
      return [];
    }
  },

  add: async (place) => {
    try {
      const places = await savedPlacesStorage.getAll();
      const newPlace = {
        ...place,
        id: place.id || `place_${Date.now()}`,
        savedAt: new Date().toISOString(),
      };
      places.unshift(newPlace);
      await AsyncStorage.setItem(KEYS.SAVED_PLACES, JSON.stringify(places));
      return newPlace;
    } catch (error) {
      console.error('Failed to add saved place:', error);
      return null;
    }
  },

  remove: async (placeId) => {
    try {
      const places = await savedPlacesStorage.getAll();
      const filtered = places.filter(p => p.id !== placeId);
      await AsyncStorage.setItem(KEYS.SAVED_PLACES, JSON.stringify(filtered));
      return true;
    } catch (error) {
      console.error('Failed to remove saved place:', error);
      return false;
    }
  },

  update: async (placeId, updates) => {
    try {
      const places = await savedPlacesStorage.getAll();
      const index = places.findIndex(p => p.id === placeId);
      if (index !== -1) {
        places[index] = { ...places[index], ...updates };
        await AsyncStorage.setItem(KEYS.SAVED_PLACES, JSON.stringify(places));
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to update saved place:', error);
      return false;
    }
  },
};

// 최근 경로
export const recentRoutesStorage = {
  getAll: async () => {
    try {
      const data = await AsyncStorage.getItem(KEYS.RECENT_ROUTES);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Failed to load recent routes:', error);
      return [];
    }
  },

  add: async (route) => {
    try {
      const routes = await recentRoutesStorage.getAll();
      const newRoute = {
        ...route,
        id: route.id || `route_${Date.now()}`,
        searchedAt: new Date().toISOString(),
      };

      // 중복 제거 (같은 출발지-도착지)
      const filtered = routes.filter(r =>
        !(r.start.latitude === route.start.latitude &&
          r.start.longitude === route.start.longitude &&
          r.end.latitude === route.end.latitude &&
          r.end.longitude === route.end.longitude)
      );

      filtered.unshift(newRoute);

      // 최대 20개까지만 저장
      const limited = filtered.slice(0, 20);
      await AsyncStorage.setItem(KEYS.RECENT_ROUTES, JSON.stringify(limited));
      return newRoute;
    } catch (error) {
      console.error('Failed to add recent route:', error);
      return null;
    }
  },

  remove: async (routeId) => {
    try {
      const routes = await recentRoutesStorage.getAll();
      const filtered = routes.filter(r => r.id !== routeId);
      await AsyncStorage.setItem(KEYS.RECENT_ROUTES, JSON.stringify(filtered));
      return true;
    } catch (error) {
      console.error('Failed to remove recent route:', error);
      return false;
    }
  },

  clear: async () => {
    try {
      await AsyncStorage.setItem(KEYS.RECENT_ROUTES, JSON.stringify([]));
      return true;
    } catch (error) {
      console.error('Failed to clear recent routes:', error);
      return false;
    }
  },
};

// 나의 제보
export const myReportsStorage = {
  getAll: async () => {
    try {
      const data = await AsyncStorage.getItem(KEYS.MY_REPORTS);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Failed to load my reports:', error);
      return [];
    }
  },

  add: async (report) => {
    try {
      const reports = await myReportsStorage.getAll();
      const newReport = {
        ...report,
        id: report.id || `report_${Date.now()}`,
        createdAt: new Date().toISOString(),
        status: report.status || 'pending',
      };
      reports.unshift(newReport);
      await AsyncStorage.setItem(KEYS.MY_REPORTS, JSON.stringify(reports));

      // 통계 업데이트
      await statsStorage.incrementReportsSubmitted();

      return newReport;
    } catch (error) {
      console.error('Failed to add report:', error);
      return null;
    }
  },

  updateStatus: async (reportId, status) => {
    try {
      const reports = await myReportsStorage.getAll();
      const index = reports.findIndex(r => r.id === reportId);
      if (index !== -1) {
        reports[index].status = status;
        await AsyncStorage.setItem(KEYS.MY_REPORTS, JSON.stringify(reports));

        // 검증된 제보 통계 업데이트
        if (status === 'verified') {
          await statsStorage.incrementReportsVerified();
        }

        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to update report status:', error);
      return false;
    }
  },
};

// 긴급 연락처
export const emergencyContactsStorage = {
  getAll: async () => {
    try {
      const data = await AsyncStorage.getItem(KEYS.EMERGENCY_CONTACTS);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Failed to load emergency contacts:', error);
      return [];
    }
  },

  add: async (contact) => {
    try {
      const contacts = await emergencyContactsStorage.getAll();

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
        shareLocation: contact.shareLocation !== undefined ? contact.shareLocation : true,
        createdAt: new Date().toISOString(),
      };

      contacts.push(newContact);
      await AsyncStorage.setItem(KEYS.EMERGENCY_CONTACTS, JSON.stringify(contacts));
      return newContact;
    } catch (error) {
      console.error('Failed to add emergency contact:', error);
      return null;
    }
  },

  remove: async (contactId) => {
    try {
      const contacts = await emergencyContactsStorage.getAll();
      const filtered = contacts.filter(c => c.id !== contactId);

      // 우선순위 재조정
      const reordered = filtered.map((contact, index) => ({
        ...contact,
        priority: index + 1,
      }));

      await AsyncStorage.setItem(KEYS.EMERGENCY_CONTACTS, JSON.stringify(reordered));
      return true;
    } catch (error) {
      console.error('Failed to remove emergency contact:', error);
      return false;
    }
  },

  update: async (contactId, updates) => {
    try {
      const contacts = await emergencyContactsStorage.getAll();
      const index = contacts.findIndex(c => c.id === contactId);
      if (index !== -1) {
        contacts[index] = { ...contacts[index], ...updates };
        await AsyncStorage.setItem(KEYS.EMERGENCY_CONTACTS, JSON.stringify(contacts));
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to update emergency contact:', error);
      return false;
    }
  },

  reorder: async (contactIds) => {
    try {
      const contacts = await emergencyContactsStorage.getAll();
      const reordered = contactIds.map((id, index) => {
        const contact = contacts.find(c => c.id === id);
        return { ...contact, priority: index + 1 };
      });
      await AsyncStorage.setItem(KEYS.EMERGENCY_CONTACTS, JSON.stringify(reordered));
      return true;
    } catch (error) {
      console.error('Failed to reorder emergency contacts:', error);
      return false;
    }
  },
};

// 설정
export const settingsStorage = {
  get: async () => {
    try {
      const data = await AsyncStorage.getItem(KEYS.SETTINGS);
      return data ? JSON.parse(data) : {
        notifications: true,
        language: 'ko',
        autoSync: true,
      };
    } catch (error) {
      console.error('Failed to load settings:', error);
      return null;
    }
  },

  save: async (settings) => {
    try {
      await AsyncStorage.setItem(KEYS.SETTINGS, JSON.stringify(settings));
      return true;
    } catch (error) {
      console.error('Failed to save settings:', error);
      return false;
    }
  },
};

// 통계
export const statsStorage = {
  get: async () => {
    try {
      const data = await AsyncStorage.getItem(KEYS.STATS);
      return data ? JSON.parse(data) : {
        reportsSubmitted: 0,
        reportsVerified: 0,
        routesCalculated: 0,
      };
    } catch (error) {
      console.error('Failed to load stats:', error);
      return {
        reportsSubmitted: 0,
        reportsVerified: 0,
        routesCalculated: 0,
      };
    }
  },

  incrementReportsSubmitted: async () => {
    try {
      const stats = await statsStorage.get();
      stats.reportsSubmitted += 1;
      await AsyncStorage.setItem(KEYS.STATS, JSON.stringify(stats));
      return stats;
    } catch (error) {
      console.error('Failed to increment reports submitted:', error);
      return null;
    }
  },

  incrementReportsVerified: async () => {
    try {
      const stats = await statsStorage.get();
      stats.reportsVerified += 1;
      await AsyncStorage.setItem(KEYS.STATS, JSON.stringify(stats));
      return stats;
    } catch (error) {
      console.error('Failed to increment reports verified:', error);
      return null;
    }
  },

  incrementRoutesCalculated: async () => {
    try {
      const stats = await statsStorage.get();
      stats.routesCalculated += 1;
      await AsyncStorage.setItem(KEYS.STATS, JSON.stringify(stats));
      return stats;
    } catch (error) {
      console.error('Failed to increment routes calculated:', error);
      return null;
    }
  },

  reset: async () => {
    try {
      await AsyncStorage.setItem(KEYS.STATS, JSON.stringify({
        reportsSubmitted: 0,
        reportsVerified: 0,
        routesCalculated: 0,
      }));
      return true;
    } catch (error) {
      console.error('Failed to reset stats:', error);
      return false;
    }
  },
};

// 전체 데이터 초기화
export const clearAllData = async () => {
  try {
    await AsyncStorage.multiRemove(Object.values(KEYS));
    return true;
  } catch (error) {
    console.error('Failed to clear all data:', error);
    return false;
  }
};

// 온보딩 완료 상태
export const onboardingStorage = {
  isCompleted: async () => {
    try {
      const completed = await AsyncStorage.getItem(KEYS.ONBOARDING_COMPLETED);
      return completed === 'true';
    } catch (error) {
      console.error('Failed to check onboarding status:', error);
      return false;
    }
  },

  setCompleted: async () => {
    try {
      await AsyncStorage.setItem(KEYS.ONBOARDING_COMPLETED, 'true');
      return true;
    } catch (error) {
      console.error('Failed to set onboarding completed:', error);
      return false;
    }
  },

  reset: async () => {
    try {
      await AsyncStorage.removeItem(KEYS.ONBOARDING_COMPLETED);
      return true;
    } catch (error) {
      console.error('Failed to reset onboarding:', error);
      return false;
    }
  },
};

// 사용자 국가 설정
export const userCountryStorage = {
  get: async () => {
    try {
      const data = await AsyncStorage.getItem(KEYS.USER_COUNTRY);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error('Failed to load user country:', error);
      return null;
    }
  },

  save: async (country) => {
    try {
      await AsyncStorage.setItem(KEYS.USER_COUNTRY, JSON.stringify(country));
      return true;
    } catch (error) {
      console.error('Failed to save user country:', error);
      return false;
    }
  },
};

// 데이터 내보내기
export const exportAllData = async () => {
  try {
    const [
      userProfile,
      savedPlaces,
      recentRoutes,
      myReports,
      emergencyContacts,
      settings,
      stats,
    ] = await Promise.all([
      userProfileStorage.get(),
      savedPlacesStorage.getAll(),
      recentRoutesStorage.getAll(),
      myReportsStorage.getAll(),
      emergencyContactsStorage.getAll(),
      settingsStorage.get(),
      statsStorage.get(),
    ]);

    return {
      userProfile,
      savedPlaces,
      recentRoutes,
      myReports,
      emergencyContacts,
      settings,
      stats,
      exportedAt: new Date().toISOString(),
    };
  } catch (error) {
    console.error('Failed to export data:', error);
    return null;
  }
};
