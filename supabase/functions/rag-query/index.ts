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
    const { action, collectionId, query, documentFile, documentName, userId } = await req.json();

    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
    const supabaseUrl = Deno.env.get('SUPABASE_URL');

    if (!serviceRoleKey || !supabaseUrl) {
      throw new Error('Missing Supabase configuration');
    }

    if (action === 'query') {
      // Simple keyword-based retrieval (placeholder for full RAG)
      if (!collectionId || !query) {
        throw new Error('Missing collection ID or query');
      }

      const startTime = Date.now();

      // Get documents from collection
      const docsResponse = await fetch(
        `${supabaseUrl}/rest/v1/knowledge_documents?collection_id=eq.${collectionId}&select=*`,
        {
          headers: {
            'Authorization': `Bearer ${serviceRoleKey}`,
            'apikey': serviceRoleKey
          }
        }
      );

      if (!docsResponse.ok) {
        throw new Error('Failed to fetch documents');
      }

      const documents = await docsResponse.json();

      // Simple keyword matching
      const queryLower = query.toLowerCase();
      const relevantDocs = documents
        .filter((doc: any) => {
          const content = (doc.content || '').toLowerCase();
          const filename = doc.filename.toLowerCase();
          return content.includes(queryLower) || filename.includes(queryLower);
        })
        .slice(0, 5);

      // Generate response based on relevant documents
      const responseText = relevantDocs.length > 0
        ? `基于知识库检索到${relevantDocs.length}个相关文档：\n\n` +
          relevantDocs.map((doc: any, i: number) => 
            `${i + 1}. ${doc.filename}\n摘要：${(doc.content || '').substring(0, 200)}...`
          ).join('\n\n')
        : '未找到相关文档，请尝试调整查询关键词。';

      const queryTimeMs = Date.now() - startTime;

      // Record query
      await fetch(`${supabaseUrl}/rest/v1/rag_queries`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${serviceRoleKey}`,
          'apikey': serviceRoleKey,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          collection_id: collectionId,
          query_text: query,
          result_count: relevantDocs.length,
          response_text: responseText,
          user_id: userId,
          query_time_ms: queryTimeMs
        })
      });

      return new Response(JSON.stringify({
        data: {
          response: responseText,
          documents: relevantDocs.map((doc: any) => ({
            id: doc.id,
            filename: doc.filename,
            content: doc.content?.substring(0, 500)
          })),
          queryTime: queryTimeMs
        }
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });

    } else if (action === 'upload') {
      // Upload document to knowledge base
      if (!collectionId || !documentFile || !documentName) {
        throw new Error('Missing required fields');
      }

      // Extract base64 data
      const base64Data = documentFile.includes(',') ? documentFile.split(',')[1] : documentFile;
      
      // Convert to text (simple extraction)
      const binaryData = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
      const decoder = new TextDecoder();
      const content = decoder.decode(binaryData);

      // Create document record
      const insertResponse = await fetch(`${supabaseUrl}/rest/v1/knowledge_documents`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${serviceRoleKey}`,
          'apikey': serviceRoleKey,
          'Content-Type': 'application/json',
          'Prefer': 'return=representation'
        },
        body: JSON.stringify({
          collection_id: collectionId,
          filename: documentName,
          file_path: `/knowledge-docs/${collectionId}/${documentName}`,
          file_size: binaryData.length,
          file_type: documentName.split('.').pop() || 'txt',
          content: content.substring(0, 10000), // Store first 10k characters
          chunk_count: Math.ceil(content.length / 500),
          upload_status: 'completed',
          uploaded_by: userId
        })
      });

      if (!insertResponse.ok) {
        const errorText = await insertResponse.text();
        throw new Error(`Failed to create document: ${errorText}`);
      }

      const documentData = await insertResponse.json();

      // Update collection document count
      await fetch(`${supabaseUrl}/rest/v1/knowledge_collections?id=eq.${collectionId}`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${serviceRoleKey}`,
          'apikey': serviceRoleKey,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          document_count: documents.length + 1,
          updated_at: new Date().toISOString()
        })
      });

      return new Response(JSON.stringify({
        data: {
          document: documentData[0]
        }
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });

    } else if (action === 'create_collection') {
      // Create new knowledge collection
      const { name, description } = await req.json();

      if (!name) {
        throw new Error('Collection name is required');
      }

      const insertResponse = await fetch(`${supabaseUrl}/rest/v1/knowledge_collections`, {
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
          collection_type: 'document',
          document_count: 0,
          is_active: true,
          created_by: userId
        })
      });

      if (!insertResponse.ok) {
        const errorText = await insertResponse.text();
        throw new Error(`Failed to create collection: ${errorText}`);
      }

      const collectionData = await insertResponse.json();

      return new Response(JSON.stringify({
        data: {
          collection: collectionData[0]
        }
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });

    } else {
      throw new Error('Invalid action');
    }

  } catch (error) {
    console.error('RAG query error:', error);

    return new Response(JSON.stringify({
      error: {
        code: 'RAG_ERROR',
        message: error.message
      }
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
});
