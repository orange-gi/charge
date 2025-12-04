import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { apiRequest } from '../lib/api';

interface User {
  id: number;
  username: string;
  email: string;
  role: 'user' | 'admin';
  firstName?: string;
  lastName?: string;
  lastLogin?: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  
  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => Promise<void>;
  initialize: () => Promise<void>;
  updateProfile: (data: Partial<User>) => Promise<void>;
}

interface LoginCredentials {
  email: string;
  password: string;
}

interface RegisterData {
  username: string;
  email: string;
  password: string;
  firstName?: string;
  lastName?: string;
}

interface BackendUser {
  id: number;
  username: string;
  email: string;
  role: 'user' | 'admin';
  first_name?: string;
  last_name?: string;
  last_login?: string;
}

interface AuthPayload {
  user: BackendUser;
  token: string;
}

function transformUser(payload: BackendUser): User {
  return {
    id: payload.id,
    username: payload.username,
    email: payload.email,
    role: payload.role,
    firstName: payload.first_name,
    lastName: payload.last_name,
    lastLogin: payload.last_login
  };
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: false,
      isAuthenticated: false,

      login: async (credentials: LoginCredentials) => {
        set({ isLoading: true });
        try {
          const response = await apiRequest<AuthPayload>('/api/auth/login', {
            method: 'POST',
            body: {
              email: credentials.email,
              password: credentials.password
            }
          });
          
          set({
            user: transformUser(response.user),
            token: response.token,
            isAuthenticated: true,
            isLoading: false
          });
        } catch (error: any) {
          set({ isLoading: false });
          throw new Error(error.message || '登录失败');
        }
      },

      register: async (data: RegisterData) => {
        set({ isLoading: true });
        try {
          const response = await apiRequest<AuthPayload>('/api/auth/register', {
            method: 'POST',
            body: {
              email: data.email,
              password: data.password,
              username: data.username,
              first_name: data.firstName,
              last_name: data.lastName
            }
          });

          // Auto login after registration
          set({
            user: transformUser(response.user),
            token: response.token,
            isAuthenticated: true,
            isLoading: false
          });
        } catch (error: any) {
          set({ isLoading: false });
          throw new Error(error.message || '注册失败');
        }
      },

      logout: async () => {
        const { token } = get();
        if (token) {
          try {
            await apiRequest('/api/auth/logout', {
              method: 'POST',
              token
            });
          } catch {
            // Ignore logout errors
          }
        }
        set({ user: null, token: null, isAuthenticated: false });
      },

      initialize: async () => {
        const { token } = get();
        if (!token) {
          set({ isLoading: false });
          return;
        }

        set({ isLoading: true });
        try {
          const user = await apiRequest<BackendUser>('/api/auth/me', {
            method: 'GET',
            token
          });
          
          set({ 
            user: transformUser(user), 
            isAuthenticated: true,
            isLoading: false
          });
        } catch (error) {
          set({ user: null, token: null, isAuthenticated: false });
          set({ isLoading: false });
        }
      },

      updateProfile: async (data: Partial<User>) => {
        const { user } = get();
        if (!user) throw new Error('未登录');

        try {
          // For now, just update local state
          // In production, you'd call an Edge Function to update the database
          set({ 
            user: {
              ...user,
              ...data
            }
          });
        } catch (error: any) {
          throw new Error(error.message || '更新失败');
        }
      }
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated
      })
    }
  )
);
