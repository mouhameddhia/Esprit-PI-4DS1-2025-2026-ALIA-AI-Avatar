import { useAuth0 } from '@auth0/auth0-react';
import { LogIn, LogOut, User } from 'lucide-react';

export function LoginButton() {
  const { loginWithRedirect } = useAuth0();

  return (
    <button
      onClick={() => loginWithRedirect()}
      className="btn btn-primary"
    >
      <LogIn size={18} /> Login
    </button>
  );
}

export function LoginButtonPopup() {
  const { loginWithPopup } = useAuth0();

  const handleLogin = async () => {
    try {
      await loginWithPopup();
    } catch (error) {
      console.error('Login failed:', error);
    }
  };

  return (
    <button
      onClick={handleLogin}
      className="btn btn-primary"
    >
      <LogIn size={18} /> Login with Auth0
    </button>
  );
}

export function SignupButton() {
  const { loginWithRedirect } = useAuth0();

  const signup = () =>
    loginWithRedirect({
      authorizationParams: { screen_hint: 'signup' },
    });

  return (
    <button
      onClick={signup}
      className="btn btn-primary"
    >
      <User size={18} /> Sign Up
    </button>
  );
}

export function SignupButtonPopup() {
  const { loginWithPopup } = useAuth0();

  const handleSignup = async () => {
    try {
      await loginWithPopup({
        authorizationParams: { screen_hint: 'signup' },
      });
    } catch (error) {
      console.error('Signup failed:', error);
    }
  };

  return (
    <button
      onClick={handleSignup}
      className="btn btn-primary"
    >
      <User size={18} /> Sign Up with Auth0
    </button>
  );
}

export function LogoutButton() {
  const { logout } = useAuth0();

  const handleLogout = () =>
    logout({ logoutParams: { returnTo: window.location.origin } });

  return (
    <button
      onClick={handleLogout}
      className="btn btn-secondary"
    >
      <LogOut size={18} /> Logout
    </button>
  );
}

export function UserProfile() {
  const { user, isAuthenticated, isLoading } = useAuth0();

  if (isLoading) return <div>Loading...</div>;

  if (!isAuthenticated) return null;

  return (
    <div className="user-profile">
      <h2>User Profile</h2>
      <p><strong>Email:</strong> {user.email}</p>
      <p><strong>Name:</strong> {user.name}</p>
      {user.picture && <img src={user.picture} alt={user.name} />}
      <pre>{JSON.stringify(user, null, 2)}</pre>
    </div>
  );
}
