/**
 * 로깅 유틸리티
 * 개발 모드에서만 console 출력
 */

const isDev = __DEV__;

export const logger = {
  log: (...args) => {
    if (isDev) {
      console.log(...args);
    }
  },

  info: (...args) => {
    if (isDev) {
      console.info(...args);
    }
  },

  warn: (...args) => {
    if (isDev) {
      console.warn(...args);
    }
  },

  error: (...args) => {
    // 에러는 항상 로깅 (프로덕션에서도 중요)
    console.error(...args);
  },

  debug: (tag, ...args) => {
    if (isDev) {
      console.log(`[${tag}]`, ...args);
    }
  },
};

export default logger;
