Deno.serve(async (req) => {
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    'Access-Control-Max-Age': '86400',
    'Access-Control-Allow-Credentials': 'false'
  };

  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  try {
    const { analysisId, userId } = await req.json();

    if (!analysisId || !userId) {
      throw new Error('Missing required fields');
    }

    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
    const supabaseUrl = Deno.env.get('SUPABASE_URL');

    if (!serviceRoleKey || !supabaseUrl) {
      throw new Error('Missing Supabase configuration');
    }

    // Update status to processing
    await updateAnalysisStatus(supabaseUrl, serviceRoleKey, analysisId, 'processing', 10);

    // Simulate file validation
    await new Promise(resolve => setTimeout(resolve, 1000));
    await updateAnalysisStatus(supabaseUrl, serviceRoleKey, analysisId, 'processing', 20);

    // Simulate message parsing
    await new Promise(resolve => setTimeout(resolve, 2000));
    await updateAnalysisStatus(supabaseUrl, serviceRoleKey, analysisId, 'processing', 40);

    // Generate mock analysis data
    const mockData = generateMockAnalysisData();
    
    // Simulate flow control model
    await new Promise(resolve => setTimeout(resolve, 1000));
    await updateAnalysisStatus(supabaseUrl, serviceRoleKey, analysisId, 'processing', 60);

    // Simulate RAG retrieval
    await new Promise(resolve => setTimeout(resolve, 500));
    await updateAnalysisStatus(supabaseUrl, serviceRoleKey, analysisId, 'processing', 80);

    // Generate analysis results
    const analysisResults = [
      {
        analysis_id: analysisId,
        result_type: 'summary',
        title: '充电分析总结',
        content: '检测到充电系统电流异常波动，可能影响充电效率。充电过程整体正常，但存在3次电流突降情况，建议检查充电连接器接触情况。',
        confidence_score: 0.85,
        metadata: JSON.stringify({ category: 'overview' })
      },
      {
        analysis_id: analysisId,
        result_type: 'finding',
        title: '关键发现1',
        content: '充电过程中出现3次电流突降，持续时间约2-3秒',
        confidence_score: 0.92,
        metadata: JSON.stringify({ 
          category: 'anomaly',
          signal: 'BMS_BattCurrt',
          timestamps: ['2025-11-19T10:15:23', '2025-11-19T10:18:45', '2025-11-19T10:22:10']
        })
      },
      {
        analysis_id: analysisId,
        result_type: 'finding',
        title: '关键发现2',
        content: '电池SOC显示正常充电曲线，从20%充至85%',
        confidence_score: 0.95,
        metadata: JSON.stringify({ 
          category: 'normal',
          signal: 'BMS_PackSOCDisp'
        })
      },
      {
        analysis_id: analysisId,
        result_type: 'finding',
        title: '关键发现3',
        content: 'BMS状态切换频繁，共发生15次状态变化',
        confidence_score: 0.78,
        metadata: JSON.stringify({ 
          category: 'anomaly',
          signal: 'BMS_DCChrgSt'
        })
      },
      {
        analysis_id: analysisId,
        result_type: 'recommendation',
        title: '建议措施1',
        content: '检查充电连接器接触情况，确保接触良好',
        confidence_score: 0.88,
        metadata: JSON.stringify({ priority: 'high' })
      },
      {
        analysis_id: analysisId,
        result_type: 'recommendation',
        title: '建议措施2',
        content: '监控温度传感器数据，确保充电温度在正常范围内',
        confidence_score: 0.82,
        metadata: JSON.stringify({ priority: 'medium' })
      },
      {
        analysis_id: analysisId,
        result_type: 'recommendation',
        title: '建议措施3',
        content: '优化充电策略参数，减少状态切换频率',
        confidence_score: 0.75,
        metadata: JSON.stringify({ priority: 'medium' })
      },
      {
        analysis_id: analysisId,
        result_type: 'technical',
        title: '技术细节',
        content: JSON.stringify(mockData),
        confidence_score: 0.90,
        metadata: JSON.stringify({ category: 'data' })
      }
    ];

    // Insert analysis results
    for (const result of analysisResults) {
      await fetch(`${supabaseUrl}/rest/v1/analysis_results`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${serviceRoleKey}`,
          'apikey': serviceRoleKey,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(result)
      });
    }

    // Update analysis to completed
    await fetch(`${supabaseUrl}/rest/v1/charging_analyses?id=eq.${analysisId}`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${serviceRoleKey}`,
        'apikey': serviceRoleKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        status: 'completed',
        progress: 100,
        completed_at: new Date().toISOString(),
        result_data: JSON.stringify({
          summary: '分析完成',
          riskLevel: 'medium',
          completedAt: new Date().toISOString()
        })
      })
    });

    return new Response(JSON.stringify({
      data: {
        status: 'completed',
        results: analysisResults
      }
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Analysis error:', error);

    // Update analysis status to failed
    try {
      const { analysisId } = await req.json();
      const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
      const supabaseUrl = Deno.env.get('SUPABASE_URL');
      
      if (analysisId && serviceRoleKey && supabaseUrl) {
        await fetch(`${supabaseUrl}/rest/v1/charging_analyses?id=eq.${analysisId}`, {
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
    } catch (e) {
      console.error('Failed to update error status:', e);
    }

    return new Response(JSON.stringify({
      error: {
        code: 'ANALYSIS_FAILED',
        message: error.message
      }
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
});

async function updateAnalysisStatus(
  supabaseUrl: string, 
  serviceRoleKey: string, 
  analysisId: number, 
  status: string, 
  progress: number
) {
  await fetch(`${supabaseUrl}/rest/v1/charging_analyses?id=eq.${analysisId}`, {
    method: 'PATCH',
    headers: {
      'Authorization': `Bearer ${serviceRoleKey}`,
      'apikey': serviceRoleKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      status,
      progress,
      ...(progress === 10 && { started_at: new Date().toISOString() })
    })
  });
}

function generateMockAnalysisData() {
  // Generate mock charging data
  const dataPoints = [];
  const startTime = Date.now() - 1000 * 60 * 30; // 30 minutes ago
  
  for (let i = 0; i < 100; i++) {
    const time = startTime + i * 1000 * 18; // 18 seconds interval
    const soc = Math.min(20 + (i / 100) * 65, 85); // SOC from 20% to 85%
    const current = 25 + Math.random() * 10 - 5; // Current around 25A
    const voltage = 350 + Math.random() * 20 - 10; // Voltage around 350V
    
    dataPoints.push({
      timestamp: new Date(time).toISOString(),
      soc: parseFloat(soc.toFixed(2)),
      current: parseFloat(current.toFixed(2)),
      voltage: parseFloat(voltage.toFixed(2)),
      status: i % 20 === 0 ? 0 : 2 // Occasional status change
    });
  }

  return {
    totalRecords: dataPoints.length,
    timeRange: {
      start: dataPoints[0].timestamp,
      end: dataPoints[dataPoints.length - 1].timestamp
    },
    signalStats: {
      BMS_DCChrgSt: {
        uniqueValues: [0, 2],
        distribution: { '0': 5, '2': 95 }
      },
      BMS_BattCurrt: {
        mean: 25.3,
        std: 4.2,
        min: 15.2,
        max: 34.8
      },
      BMS_PackSOCDisp: {
        mean: 52.5,
        std: 18.7,
        min: 20.0,
        max: 85.0
      }
    },
    dataPoints: dataPoints.slice(0, 20) // Return first 20 points as sample
  };
}
