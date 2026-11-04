"use strict";

(() => {
  const auth = window.growthlabAuth;

  const overlay = document.getElementById("glAuthOverlay");
  const form = document.getElementById("glAuthForm");
  const emailInput = document.getElementById("glAuthEmail");
  const passwordInput = document.getElementById("glAuthPassword");
  const loginTab = document.getElementById("glLoginTab");
  const signupTab = document.getElementById("glSignupTab");
  const submitButton = document.getElementById("glAuthSubmit");
  const message = document.getElementById("glAuthMessage");
  const title = document.getElementById("glAuthTitle");
  const subtitle = document.getElementById("glAuthSubtitle");

  let mode = "login";
  let initialized = false;

  function showMessage(text, success = false) {
    message.textContent = text;
    message.classList.toggle("success", success);
  }

  function setMode(nextMode) {
    mode = nextMode;

    loginTab.classList.toggle("active", mode === "login");
    signupTab.classList.toggle("active", mode === "signup");

    title.textContent = mode === "login"
      ? "Welcome back"
      : "Create your account";

    subtitle.textContent = mode === "login"
      ? "Sign in to continue using GrowthLab AI."
      : "Create a free account to start using GrowthLab AI.";

    submitButton.textContent = mode === "login"
      ? "Log In"
      : "Create Account";

    passwordInput.autocomplete = mode === "login"
      ? "current-password"
      : "new-password";

    showMessage("");
  }

  function getAccountPanel() {
    let panel = document.getElementById("glAuthAccount");

    if (panel) {
      return panel;
    }

    panel = document.createElement("div");
    panel.id = "glAuthAccount";
    panel.className = "gl-auth-account";
    panel.hidden = true;

    const email = document.createElement("span");
    email.id = "glAuthAccountEmail";
    email.className = "gl-auth-account-email";

    const logout = document.createElement("button");
    logout.type = "button";
    logout.className = "gl-auth-logout";
    logout.textContent = "Log Out";

    logout.addEventListener("click", async () => {
      logout.disabled = true;

      try {
        await auth.signOut();

        sessionStorage.removeItem("growthlab_ai_session_id");

        window.location.reload();
      } catch (error) {
        logout.disabled = false;
        alert(error.message || "Unable to log out.");
      }
    });

    panel.appendChild(email);
    panel.appendChild(logout);

    const sidebar = document.querySelector(".sidebar-bottom");

    if (sidebar) {
      sidebar.prepend(panel);
    }

    return panel;
  }

  function updateUI() {
    if (!initialized) {
      return;
    }

    const isCloud = auth.cloudMode();
    const loggedIn = auth.isAuthenticated();

    overlay.hidden = !isCloud || loggedIn;

    const accountPanel = getAccountPanel();
    accountPanel.hidden = !isCloud || !loggedIn;

    if (isCloud && loggedIn) {
      document.getElementById("glAuthAccountEmail").textContent =
        auth.getUser()?.email || "Signed in";
    }
  }

  loginTab.addEventListener("click", () => setMode("login"));
  signupTab.addEventListener("click", () => setMode("signup"));

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const email = emailInput.value.trim();
    const password = passwordInput.value;

    if (!email || !password) {
      showMessage("Please enter your email and password.");
      return;
    }

    submitButton.disabled = true;
    showMessage("");

    try {
      if (mode === "signup") {
        const result = await auth.signUp(email, password);

        if (result.needsConfirmation) {
          setMode("login");
          showMessage(
            "Account created. Check your email for a confirmation link, " +
            "then return here to log in.",
            true
          );
        } else {
          updateUI();
        }
      } else {
        await auth.signIn(email, password);
        updateUI();
      }
    } catch (error) {
      showMessage(error.message || "Authentication failed.");
    } finally {
      submitButton.disabled = false;
    }
  });

  window.addEventListener("growthlab-auth-changed", updateUI);

  async function start() {
    // Block the chat while cloud authentication initializes.
    overlay.hidden = false;
    submitButton.disabled = true;
    showMessage("Checking authentication...");

    try {
      await auth.initialize();
      initialized = true;
      submitButton.disabled = false;
      showMessage("");
      updateUI();
    } catch (error) {
      initialized = false;
      overlay.hidden = false;
      submitButton.disabled = true;
      showMessage(
        error.message || "Authentication is temporarily unavailable."
      );
    }
  }

  start();
})();
