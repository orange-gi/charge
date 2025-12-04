Deno.serve(async (req) => {
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-custom-token',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, DELETE',
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

    const { action, ...params } = await req.json();

    let result;

    switch (action) {
      case 'upload_dataset':
        result = await uploadDataset(supabaseUrl, serviceRoleKey, params);
        break;
      
      case 'create_training_task':
        result = await createTrainingTask(supabaseUrl, serviceRoleKey, params);
        break;
      
      case 'start_training':
        result = await startTraining(supabaseUrl, serviceRoleKey, params);
        break;
      
      case 'get_training_status':
        result = await getTrainingStatus(supabaseUrl, serviceRoleKey, params);
        break;
      
      case 'create_model_version':
        result = await createModelVersion(supabaseUrl, serviceRoleKey, params);
        break;
      
      default:
        throw new Error('Invalid action');
    }

    return new Response(JSON.stringify({ data: result }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Training management error:', error);

    return new Response(JSON.stringify({
      error: {
        code: 'TRAINING_ERROR',
        message: error.message
      }
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
});

async function uploadDataset(supabaseUrl: string, serviceRoleKey: string, params: any) {
  const { name, description, datasetType, fileData, fileName, userId } = params;

  // Parse CSV data
  const base64Data = fileData.includes(',') ? fileData.split(',')[1] : fileData;
  const binaryData = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
  const decoder = new TextDecoder();
  const csvContent = decoder.decode(binaryData);
  
  // Count samples
  const lines = csvContent.split('\n').filter(line => line.trim());
  const sampleCount = Math.max(0, lines.length - 1); // Exclude header

  // Create dataset record
  const response = await fetch(`${supabaseUrl}/rest/v1/training_datasets`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${serviceRoleKey}`,
      'apikey': serviceRoleKey,
      'Content-Type': 'application/json',
      'Prefer': 'return=representation'
    },
    body: JSON.stringify({
      name,
      description: description || '',
      dataset_type: datasetType || 'standard',
      file_path: `/training-datasets/${userId}/${Date.now()}-${fileName}`,
      sample_count: sampleCount,
      metadata: JSON.stringify({
        fileName,
        fileSize: binaryData.length,
        uploadedAt: new Date().toISOString()
      }),
      is_public: false,
      created_by: userId
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to create dataset: ${error}`);
  }

  const data = await response.json();
  return { dataset: data[0] };
}

async function createTrainingTask(supabaseUrl: string, serviceRoleKey: string, params: any) {
  const { name, description, datasetId, modelType, hyperparameters, userId } = params;

  const response = await fetch(`${supabaseUrl}/rest/v1/training_tasks`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${serviceRoleKey}`,
      'apikey': serviceRoleKey,
      'Content-Type': 'application/json',
      'Prefer': 'return=representation'
    },
    body: JSON.stringify({
      name,
      description: description || '',
      dataset_id: datasetId,
      model_type: modelType,
      hyperparameters: JSON.stringify(hyperparameters || {}),
      status: 'pending',
      progress: 0,
      current_epoch: 0,
      total_epochs: hyperparameters?.epochs || 10,
      current_step: 0,
      total_steps: 0,
      created_by: userId
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to create training task: ${error}`);
  }

  const data = await response.json();
  return { task: data[0] };
}

async function startTraining(supabaseUrl: string, serviceRoleKey: string, params: any) {
  const { taskId } = params;

  // Update task status to running
  await fetch(`${supabaseUrl}/rest/v1/training_tasks?id=eq.${taskId}`, {
    method: 'PATCH',
    headers: {
      'Authorization': `Bearer ${serviceRoleKey}`,
      'apikey': serviceRoleKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      status: 'running',
      start_time: new Date().toISOString(),
      progress: 0
    })
  });

  // Simulate training process in background
  simulateTraining(supabaseUrl, serviceRoleKey, taskId);

  return { message: 'Training started successfully', taskId };
}

async function simulateTraining(supabaseUrl: string, serviceRoleKey: string, taskId: number) {
  // This simulates a training process
  // In production, this would trigger an actual ML training job
  
  try {
    const totalEpochs = 10;
    const stepsPerEpoch = 100;

    for (let epoch = 1; epoch <= totalEpochs; epoch++) {
      for (let step = 1; step <= stepsPerEpoch; step++) {
        const progress = ((epoch - 1) * stepsPerEpoch + step) / (totalEpochs * stepsPerEpoch) * 100;
        
        // Simulate metrics
        const loss = 2.5 * Math.exp(-epoch * 0.3) + Math.random() * 0.1;
        const accuracy = 0.6 + (epoch / totalEpochs) * 0.35 + Math.random() * 0.05;

        // Update task progress
        await fetch(`${supabaseUrl}/rest/v1/training_tasks?id=eq.${taskId}`, {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${serviceRoleKey}`,
            'apikey': serviceRoleKey,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            progress: parseFloat(progress.toFixed(2)),
            current_epoch: epoch,
            current_step: step,
            metrics: JSON.stringify({
              loss: loss.toFixed(4),
              accuracy: accuracy.toFixed(4),
              learning_rate: 0.001
            })
          })
        });

        // Record metrics
        if (step % 10 === 0) {
          await fetch(`${supabaseUrl}/rest/v1/training_metrics`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${serviceRoleKey}`,
              'apikey': serviceRoleKey,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              task_id: taskId,
              epoch,
              step,
              loss: parseFloat(loss.toFixed(4)),
              accuracy: parseFloat(accuracy.toFixed(4)),
              learning_rate: 0.001,
              gpu_memory: 4.5 + Math.random() * 2
            })
          });
        }

        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    // Mark as completed
    await fetch(`${supabaseUrl}/rest/v1/training_tasks?id=eq.${taskId}`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${serviceRoleKey}`,
        'apikey': serviceRoleKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        status: 'completed',
        progress: 100,
        end_time: new Date().toISOString(),
        model_path: `/models/task_${taskId}_final.pth`
      })
    });

  } catch (error) {
    console.error('Training simulation error:', error);
    
    await fetch(`${supabaseUrl}/rest/v1/training_tasks?id=eq.${taskId}`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${serviceRoleKey}`,
        'apikey': serviceRoleKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        status: 'failed',
        error_message: error.message
      })
    });
  }
}

async function getTrainingStatus(supabaseUrl: string, serviceRoleKey: string, params: any) {
  const { taskId } = params;

  const response = await fetch(
    `${supabaseUrl}/rest/v1/training_tasks?id=eq.${taskId}&select=*`,
    {
      headers: {
        'Authorization': `Bearer ${serviceRoleKey}`,
        'apikey': serviceRoleKey
      }
    }
  );

  if (!response.ok) {
    throw new Error('Failed to get training status');
  }

  const data = await response.json();
  return { task: data[0] || null };
}

async function createModelVersion(supabaseUrl: string, serviceRoleKey: string, params: any) {
  const { name, modelType, version, modelPath, config, metrics, userId } = params;

  const response = await fetch(`${supabaseUrl}/rest/v1/model_versions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${serviceRoleKey}`,
      'apikey': serviceRoleKey,
      'Content-Type': 'application/json',
      'Prefer': 'return=representation'
    },
    body: JSON.stringify({
      name,
      model_type: modelType,
      version,
      model_path: modelPath,
      config: JSON.stringify(config || {}),
      metrics: JSON.stringify(metrics || {}),
      is_active: false,
      is_default: false,
      created_by: userId
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to create model version: ${error}`);
  }

  const data = await response.json();
  return { modelVersion: data[0] };
}
