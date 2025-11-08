/**
 * 지원 국가 목록
 * 각 국가별 중심 좌표와 정보
 */

export const COUNTRIES = [
  {
    code: 'SS',
    name: '남수단 (South Sudan)',
    nameEn: 'South Sudan',
    nameLocal: 'South Sudan',
    flag: '🇸🇸',
    center: {
      latitude: 4.8594,
      longitude: 31.5713,
      city: 'Juba',
    },
    zoom: 6,
  },
  {
    code: 'KE',
    name: '케냐 (Kenya)',
    nameEn: 'Kenya',
    nameLocal: 'Kenya',
    flag: '🇰🇪',
    center: {
      latitude: -1.2864,
      longitude: 36.8172,
      city: 'Nairobi',
    },
    zoom: 6,
  },
  {
    code: 'UG',
    name: '우간다 (Uganda)',
    nameEn: 'Uganda',
    nameLocal: 'Uganda',
    flag: '🇺🇬',
    center: {
      latitude: 0.3476,
      longitude: 32.5825,
      city: 'Kampala',
    },
    zoom: 7,
  },
  {
    code: 'ET',
    name: '에티오피아 (Ethiopia)',
    nameEn: 'Ethiopia',
    nameLocal: 'ኢትዮጵያ',
    flag: '🇪🇹',
    center: {
      latitude: 9.0320,
      longitude: 38.7469,
      city: 'Addis Ababa',
    },
    zoom: 6,
  },
  {
    code: 'SO',
    name: '소말리아 (Somalia)',
    nameEn: 'Somalia',
    nameLocal: 'Soomaaliya',
    flag: '🇸🇴',
    center: {
      latitude: 2.0469,
      longitude: 45.3182,
      city: 'Mogadishu',
    },
    zoom: 6,
  },
  {
    code: 'CD',
    name: '콩고민주공화국 (DR Congo)',
    nameEn: 'Democratic Republic of the Congo',
    nameLocal: 'RD Congo',
    flag: '🇨🇩',
    center: {
      latitude: -4.3276,
      longitude: 15.3136,
      city: 'Kinshasa',
    },
    zoom: 5,
  },
  {
    code: 'CF',
    name: '중앙아프리카공화국 (CAR)',
    nameEn: 'Central African Republic',
    nameLocal: 'Centrafrique',
    flag: '🇨🇫',
    center: {
      latitude: 4.3947,
      longitude: 18.5582,
      city: 'Bangui',
    },
    zoom: 6,
  },
  {
    code: 'SD',
    name: '수단 (Sudan)',
    nameEn: 'Sudan',
    nameLocal: 'السودان',
    flag: '🇸🇩',
    center: {
      latitude: 15.5007,
      longitude: 32.5599,
      city: 'Khartoum',
    },
    zoom: 5,
  },
  {
    code: 'YE',
    name: '예멘 (Yemen)',
    nameEn: 'Yemen',
    nameLocal: 'اليمن',
    flag: '🇾🇪',
    center: {
      latitude: 15.5527,
      longitude: 48.5164,
      city: 'Sana\'a',
    },
    zoom: 6,
  },
  {
    code: 'SY',
    name: '시리아 (Syria)',
    nameEn: 'Syria',
    nameLocal: 'سوريا',
    flag: '🇸🇾',
    center: {
      latitude: 33.5138,
      longitude: 36.2765,
      city: 'Damascus',
    },
    zoom: 7,
  },
  {
    code: 'IQ',
    name: '이라크 (Iraq)',
    nameEn: 'Iraq',
    nameLocal: 'العراق',
    flag: '🇮🇶',
    center: {
      latitude: 33.3152,
      longitude: 44.3661,
      city: 'Baghdad',
    },
    zoom: 6,
  },
  {
    code: 'AF',
    name: '아프가니스탄 (Afghanistan)',
    nameEn: 'Afghanistan',
    nameLocal: 'افغانستان',
    flag: '🇦🇫',
    center: {
      latitude: 34.5553,
      longitude: 69.2075,
      city: 'Kabul',
    },
    zoom: 6,
  },
];

/**
 * 국가 코드로 국가 정보 찾기
 */
export const getCountryByCode = (code) => {
  return COUNTRIES.find(country => country.code === code);
};

/**
 * 기본 국가 (남수단)
 */
export const DEFAULT_COUNTRY = COUNTRIES[0];
