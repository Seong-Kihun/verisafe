'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAuthStore } from '@/lib/stores/auth-store';

interface MenuItem {
  label: string;
  href: string;
  icon: string;
  roles: ('mapper' | 'admin')[];
}

const menuItems: MenuItem[] = [
  {
    label: '지도 편집',
    href: '/mapper',
    icon: '🗺️',
    roles: ['mapper', 'admin'],
  },
  {
    label: '내 기여',
    href: '/mapper/contributions',
    icon: '📝',
    roles: ['mapper', 'admin'],
  },
  {
    label: '검수 대기열',
    href: '/reviewer',
    icon: '✓',
    roles: ['admin'],
  },
  {
    label: '검수 대시보드',
    href: '/reviewer/dashboard',
    icon: '📊',
    roles: ['admin'],
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { user, logout } = useAuthStore();

  const filteredMenuItems = menuItems.filter((item) =>
    user?.role ? item.roles.includes(user.role) : false
  );

  return (
    <aside className="w-64 bg-gray-900 text-white flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-xl font-bold">VeriSafe Portal</h1>
        <p className="text-sm text-gray-400 mt-1">
          {user?.role === 'admin' ? '관리자' : '매퍼'}
        </p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {filteredMenuItems.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + '/');

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-primary text-white'
                  : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              }`}
            >
              <span className="text-xl">{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* User Info & Logout */}
      <div className="p-4 border-t border-gray-700">
        <div className="mb-3">
          <p className="text-sm font-medium">{user?.username}</p>
          <p className="text-xs text-gray-400">{user?.email}</p>
        </div>
        <button
          onClick={() => {
            logout();
            window.location.href = '/login';
          }}
          className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors"
        >
          로그아웃
        </button>
      </div>
    </aside>
  );
}
