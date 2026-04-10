import { useEffect } from 'react';

/**
 * Hook to automatically finalize a chat session when the user leaves the page/closes tab.
 * 
 * @param {string} sessionId - The current session ID
 * @param {string} apiBase - Base API URL (default: http://localhost:8000)
 * @param {string} sessionStorageKey - Session storage key to clear (optional)
 */
export const useSessionFinalize = (sessionId, apiBase = 'http://localhost:8000', sessionStorageKey = null) => {
  useEffect(() => {
    if (!sessionId) return;

    const handleBeforeUnload = async (event) => {
      const token = localStorage.getItem('token');
      if (!token) return;

      try {
        // Attempt to finalize the session
        await fetch(`${apiBase}/chat/sessions/${sessionId}/finalize`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
          // Use keepalive to ensure request completes even if page unloads
          keepalive: true,
        });
      } catch (error) {
        // Silently fail - we can't show errors during unload
        console.error('Failed to finalize session on page unload:', error);
      }

      // Clear session storage if key provided
      if (sessionStorageKey) {
        sessionStorage.removeItem(sessionStorageKey);
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [sessionId, apiBase, sessionStorageKey]);
};
