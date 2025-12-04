import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { callEdgeFunction } from '../lib/supabase';

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
  logout: () => void;
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
          const response = await callEdgeFunction('userAuth', {
            action: 'login',
            email: credentials.email,
            password: credentials.password
          });

          const { user, token } = response.data;
          
          set({
            user,
            token,
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
          const response = await callEdgeFunction('userAuth', {
            action: 'register',
            email: data.email,
            password: data.password,
            username: data.username,
            firstName: data.firstName,
            lastName: data.lastName
          });

          const { user, token } = response.data;

          // Auto login after registration
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false
          });
        } catch (error: any) {
          set({ isLoading: false });
          throw new Error(error.message || '注册失败');
        }
      },

      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false
        });
      },

      initialize: async () => {
        const { token } = get();
        if (!token) {
          set({ isLoading: false });
          return;
        }

        set({ isLoading: true });
        try {
          const response = await callEdgeFunction('userAuth', {
            action: 'verify'
          }, token);

          const { user } = response.data;
          
          set({ 
            user, 
            isAuthenticated: true,
            isLoading: false
          });
        } catch (error) {
          get().logout();
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
