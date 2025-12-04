Deno.serve(async (req) => {
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, PUT, DELETE, PATCH',
    'Access-Control-Max-Age': '86400',
    'Access-Control-Allow-Credentials': 'false'
  };

  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  try {
    const { action, email, password, username, firstName, lastName } = await req.json();
    
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
    const supabaseUrl = Deno.env.get('SUPABASE_URL');
    
    if (!serviceRoleKey || !supabaseUrl) {
      throw new Error('Missing Supabase configuration');
    }

    if (action === 'register') {
      // 1. Create user in users table
      const hashedPassword = await hashPassword(password);
      
      const insertResponse = await fetch(`${supabaseUrl}/rest/v1/users`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${serviceRoleKey}`,
          'apikey': serviceRoleKey,
          'Content-Type': 'application/json',
          'Prefer': 'return=representation'
        },
        body: JSON.stringify({
          username: username || email.split('@')[0],
          email,
          password_hash: hashedPassword,
          first_name: firstName || null,
          last_name: lastName || null,
          role: 'user',
          is_active: true
        })
      });

      if (!insertResponse.ok) {
        const error = await insertResponse.text();
        throw new Error(`Failed to create user: ${error}`);
      }

      const userData = await insertResponse.json();
      const user = userData[0];

      // 2. Create session token
      const token = await createSessionToken(user.id);
      const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000); // 7 days

      const sessionResponse = await fetch(`${supabaseUrl}/rest/v1/user_sessions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${serviceRoleKey}`,
          'apikey': serviceRoleKey,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: user.id,
          token_hash: token,
          expires_at: expiresAt.toISOString()
        })
      });

      if (!sessionResponse.ok) {
        console.error('Failed to create session, but user created');
      }

      return new Response(JSON.stringify({
        data: {
          user: {
            id: user.id,
            email: user.email,
            username: user.username,
            firstName: user.first_name,
            lastName: user.last_name,
            role: user.role
          },
          token
        }
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });

    } else if (action === 'login') {
      // 1. Find user by email
      const userResponse = await fetch(
        `${supabaseUrl}/rest/v1/users?email=eq.${encodeURIComponent(email)}&select=*`,
        {
          headers: {
            'Authorization': `Bearer ${serviceRoleKey}`,
            'apikey': serviceRoleKey
          }
        }
      );

      if (!userResponse.ok) {
        throw new Error('Failed to query user');
      }

      const users = await userResponse.json();
      
      if (!users || users.length === 0) {
        throw new Error('Invalid email or password');
      }

      const user = users[0];

      if (!user.is_active) {
        throw new Error('Account is disabled');
      }

      // 2. Verify password
      const passwordValid = await verifyPassword(password, user.password_hash);
      
      if (!passwordValid) {
        throw new Error('Invalid email or password');
      }

      // 3. Update last login
      await fetch(`${supabaseUrl}/rest/v1/users?id=eq.${user.id}`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${serviceRoleKey}`,
          'apikey': serviceRoleKey,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          last_login: new Date().toISOString()
        })
      });

      // 4. Create session token
      const token = await createSessionToken(user.id);
      const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);

      await fetch(`${supabaseUrl}/rest/v1/user_sessions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${serviceRoleKey}`,
          'apikey': serviceRoleKey,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: user.id,
          token_hash: token,
          expires_at: expiresAt.toISOString()
        })
      });

      return new Response(JSON.stringify({
        data: {
          user: {
            id: user.id,
            email: user.email,
            username: user.username,
            firstName: user.first_name,
            lastName: user.last_name,
            role: user.role
          },
          token
        }
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });

    } else if (action === 'verify') {
      // Verify token
      const token = req.headers.get('authorization')?.replace('Bearer ', '');
      
      if (!token) {
        throw new Error('No token provided');
      }

      // Check session
      const sessionResponse = await fetch(
        `${supabaseUrl}/rest/v1/user_sessions?token_hash=eq.${encodeURIComponent(token)}&select=*,users(*)`,
        {
          headers: {
            'Authorization': `Bearer ${serviceRoleKey}`,
            'apikey': serviceRoleKey
          }
        }
      );

      if (!sessionResponse.ok) {
        throw new Error('Session lookup failed');
      }

      const sessions = await sessionResponse.json();
      
      if (!sessions || sessions.length === 0) {
        throw new Error('Invalid or expired session');
      }

      const session = sessions[0];
      
      if (new Date(session.expires_at) < new Date()) {
        throw new Error('Session expired');
      }

      // Get user details
      const userResponse = await fetch(
        `${supabaseUrl}/rest/v1/users?id=eq.${session.user_id}&select=*`,
        {
          headers: {
            'Authorization': `Bearer ${serviceRoleKey}`,
            'apikey': serviceRoleKey
          }
        }
      );

      const users = await userResponse.json();
      const user = users[0];

      return new Response(JSON.stringify({
        data: {
          user: {
            id: user.id,
            email: user.email,
            username: user.username,
            firstName: user.first_name,
            lastName: user.last_name,
            role: user.role
          }
        }
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });

    } else {
      throw new Error('Invalid action');
    }

  } catch (error) {
    console.error('Auth error:', error);
    
    return new Response(JSON.stringify({
      error: {
        code: 'AUTH_ERROR',
        message: error.message
      }
    }), {
      status: 400,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
});

// Simple hash function using Web Crypto API
async function hashPassword(password: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(password);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

async function verifyPassword(password: string, hash: string): Promise<boolean> {
  const computedHash = await hashPassword(password);
  return computedHash === hash;
}

async function createSessionToken(userId: number): Promise<string> {
  const randomValues = new Uint8Array(32);
  crypto.getRandomValues(randomValues);
  const token = Array.from(randomValues)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return `${userId}-${token}`;
}
