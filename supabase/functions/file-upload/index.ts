Deno.serve(async (req) => {
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-custom-token',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    'Access-Control-Max-Age': '86400',
    'Access-Control-Allow-Credentials': 'false'
  };

  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  try {
    // Get user token from custom header
    const token = req.headers.get('x-custom-token');
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
    const supabaseUrl = Deno.env.get('SUPABASE_URL');

    if (!serviceRoleKey || !supabaseUrl) {
      throw new Error('Missing Supabase configuration');
    }

    // Verify custom token if provided
    if (token) {
      const sessionResponse = await fetch(
        `${supabaseUrl}/rest/v1/user_sessions?token_hash=eq.${encodeURIComponent(token)}&select=*`,
        {
          headers: {
            'Authorization': `Bearer ${serviceRoleKey}`,
            'apikey': serviceRoleKey
          }
        }
      );

      if (!sessionResponse.ok) {
        return new Response(JSON.stringify({
          error: { code: 'UNAUTHORIZED', message: 'Invalid token' }
        }), {
          status: 401,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }

      const sessions = await sessionResponse.json();
      if (!sessions || sessions.length === 0 || new Date(sessions[0].expires_at) < new Date()) {
        return new Response(JSON.stringify({
          error: { code: 'UNAUTHORIZED', message: 'Token expired or invalid' }
        }), {
          status: 401,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }
    }

    const { fileData, fileName, fileSize, userId, analysisName, description } = await req.json();

    if (!fileData || !fileName || !userId) {
      throw new Error('Missing required fields');
    }

    // Extract base64 data
    const base64Data = fileData.includes(',') ? fileData.split(',')[1] : fileData;
    const mimeType = fileData.includes(',') ? fileData.split(';')[0].split(':')[1] : 'application/octet-stream';

    // Convert base64 to binary
    const binaryData = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));

    // Generate storage path
    const timestamp = Date.now();
    const storagePath = `${userId}/${timestamp}-${fileName}`;

    // Upload to Supabase Storage
    const uploadResponse = await fetch(`${supabaseUrl}/storage/v1/object/charging-files/${storagePath}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${serviceRoleKey}`,
        'Content-Type': mimeType,
        'x-upsert': 'true'
      },
      body: binaryData
    });

    if (!uploadResponse.ok) {
      const errorText = await uploadResponse.text();
      // If bucket doesn't exist, store file path as placeholder
      console.warn('Upload failed, using placeholder:', errorText);
    }

    // Get public URL
    const publicUrl = `${supabaseUrl}/storage/v1/object/public/charging-files/${storagePath}`;

    // Create analysis record
    const insertResponse = await fetch(`${supabaseUrl}/rest/v1/charging_analyses`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${serviceRoleKey}`,
        'apikey': serviceRoleKey,
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
      },
      body: JSON.stringify({
        name: analysisName || fileName,
        description: description || '',
        file_path: publicUrl,
        file_size: fileSize || binaryData.length,
        file_type: fileName.split('.').pop() || 'unknown',
        status: 'pending',
        progress: 0,
        user_id: userId
      })
    });

    if (!insertResponse.ok) {
      const errorText = await insertResponse.text();
      throw new Error(`Failed to create analysis record: ${errorText}`);
    }

    const analysisData = await insertResponse.json();
    const analysis = analysisData[0];

    return new Response(JSON.stringify({
      data: {
        analysisId: analysis.id,
        publicUrl,
        analysis
      }
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('File upload error:', error);

    return new Response(JSON.stringify({
      error: {
        code: 'UPLOAD_FAILED',
        message: error.message
      }
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
});
