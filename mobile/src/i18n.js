/**
 * i18n 다국어 지원 설정
 * 한국어, 영어, 스페인어, 프랑스어, 포르투갈어, 스와힐리어 지원
 */

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';

// 번역 파일 import
import ko from './locales/ko.json';
import en from './locales/en.json';
import es from './locales/es.json';
import fr from './locales/fr.json';
import pt from './locales/pt.json';
import sw from './locales/sw.json';

const LANGUAGE_KEY = '@verisafe:language';

// AsyncStorage에서 저장된 언어 가져오기
const getStoredLanguage = async () => {
  try {
    const language = await AsyncStorage.getItem(LANGUAGE_KEY);
    return language || 'ko'; // 기본값: 한국어
  } catch (error) {
    console.error('Failed to load language:', error);
    return 'ko';
  }
};

// AsyncStorage에 언어 저장하기
export const setLanguage = async (language) => {
  try {
    await AsyncStorage.setItem(LANGUAGE_KEY, language);
    await i18n.changeLanguage(language);
    return true;
  } catch (error) {
    console.error('Failed to save language:', error);
    return false;
  }
};

// i18n 초기화
const initI18n = async () => {
  const storedLanguage = await getStoredLanguage();

  i18n
    .use(initReactI18next)
    .init({
      resources: {
        ko: { translation: ko },
        en: { translation: en },
        es: { translation: es },
        fr: { translation: fr },
        pt: { translation: pt },
        sw: { translation: sw },
      },
      lng: storedLanguage,
      fallbackLng: 'ko',
      compatibilityJSON: 'v3', // React Native compatibility
      interpolation: {
        escapeValue: false, // React already escapes
      },
      react: {
        useSuspense: false, // Disable suspense for React Native
      },
    });
};

// 앱 시작 시 초기화
initI18n();

export default i18n;
