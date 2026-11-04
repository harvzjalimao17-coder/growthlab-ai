"use strict";

window.growthlabAuth = (() => {
  const STORAGE_KEY = "growthlab_ai_auth";
  let config = null;
  let session = null;
  let refreshPromise = null;

  function readSession() {
    try {
      return JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
    } catch {
      return null;
    }
  }

  function saveSession(data) {
    session = data;
    if (data) {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } else {
      sessionStorage.removeItem(STORAGE_KEY);
    }
  }

  function notify() {
    window.dispatchEvent(new CustomEvent("growthlab-auth-changed", {
      detail: {
        authenticated: Boolean(session?.access_token),
        email: session?.user?.email || ""
      }
    }));
  }

  async function initialize() {
    const response = await fetch("/auth/config", {
      cache: "no-store"
    });

    if (!response.ok) {
      throw new Error("Authentication configuration unavailable.");
    }

    config = await response.json();

    if (!config.cloud_mode) {
      saveSession(null);
      notify();
      return { cloudMode: false };
    }

    if (!config.supabase_url || !config.supabase_publishable_key) {
      throw new Error("Supabase authentication is not configured.");
    }

    session = readSession();

    if (session?.refresh_token) {
      try {
        await getAccessToken();
      } catch {
        saveSession(null);
      }
    } else {
      saveSession(null);
    }

    notify();

    return {
      cloudMode: true,
      authenticated: Boolean(session?.access_token)
    };
  }

  function cloudMode() {
    return config?.cloud_mode === true;
  }

  function apiUrl(path) {
    return config.supabase_url.replace(/\/+$/, "") + "/auth/v1" + path;
  }

  async function authRequest(path, body) {
    if (!config?.cloud_mode) {
      throw new Error("Cloud authentication is not enabled.");
    }

    const response = await fetch(apiUrl(path), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "apikey": config.supabase_publishable_key
      },
      body: JSON.stringify(body)
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(
        data.msg ||
        data.error_description ||
        data.message ||
        data.error ||
        "Authentication failed."
      );
    }

    return data;
  }

  function storeAuthResponse(data) {
    if (!data.access_token || !data.refresh_token) {
      throw new Error("No active session was returned.");
    }

    saveSession({
      access_token: data.access_token,
      refresh_token: data.refresh_token,
      expires_at: Math.floor(Date.now() / 1000) +
        Number(data.expires_in || 3600),
      user: data.user || session?.user || null
    });

    notify();
  }

  async function signIn(email, password) {
    const data = await authRequest(
      "/token?grant_type=password",
      { email, password }
    );

    storeAuthResponse(data);
    return data;
  }

  async function signUp(email, password) {
    const data = await authRequest("/signup", {
      email,
      password
    });

    if (data.access_token) {
      storeAuthResponse(data);
    }

    return {
      needsConfirmation: !data.access_token
    };
  }

  async function refresh() {
    if (!session?.refresh_token) {
      throw new Error("Please log in.");
    }

    if (!refreshPromise) {
      refreshPromise = authRequest(
        "/token?grant_type=refresh_token",
        { refresh_token: session.refresh_token }
      ).then(data => {
        storeAuthResponse(data);
        return data.access_token;
      }).catch(error => {
        saveSession(null);
        notify();
        throw error;
      }).finally(() => {
        refreshPromise = null;
      });
    }

    return refreshPromise;
  }

  async function getAccessToken() {
    if (!cloudMode()) {
      return null;
    }

    if (!session?.access_token) {
      throw new Error("Please log in to use GrowthLab AI.");
    }

    const now = Math.floor(Date.now() / 1000);

    if (session.expires_at && session.expires_at <= now + 60) {
      return refresh();
    }

    return session.access_token;
  }

  async function signOut() {
    const token = session?.access_token;

    saveSession(null);
    notify();

    if (token && config?.cloud_mode) {
      try {
        await fetch(apiUrl("/logout"), {
          method: "POST",
          headers: {
            "apikey": config.supabase_publishable_key,
            "Authorization": "Bearer " + token
          }
        });
      } catch {
        // Local session has already been cleared.
      }
    }
  }

  function getUser() {
    return session?.user || null;
  }

  function isAuthenticated() {
    return Boolean(session?.access_token);
  }

  return {
    initialize,
    cloudMode,
    signIn,
    signUp,
    signOut,
    getAccessToken,
    getUser,
    isAuthenticated
  };
})();
