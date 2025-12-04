-- Migration: enable_rls_and_policies
-- Created at: 1763537900

-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE charging_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- Users table policies
CREATE POLICY "users_select_all" ON users FOR SELECT USING (true);
CREATE POLICY "users_insert" ON users FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "users_update" ON users FOR UPDATE USING (auth.role() IN ('anon', 'service_role'));

-- Charging analyses policies
CREATE POLICY "charging_analyses_select" ON charging_analyses FOR SELECT USING (true);
CREATE POLICY "charging_analyses_insert" ON charging_analyses FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "charging_analyses_update" ON charging_analyses FOR UPDATE USING (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "charging_analyses_delete" ON charging_analyses FOR DELETE USING (auth.role() IN ('anon', 'service_role'));

-- Analysis results policies
CREATE POLICY "analysis_results_select" ON analysis_results FOR SELECT USING (true);
CREATE POLICY "analysis_results_insert" ON analysis_results FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "analysis_results_update" ON analysis_results FOR UPDATE USING (auth.role() IN ('anon', 'service_role'));

-- Knowledge collections policies
CREATE POLICY "knowledge_collections_select" ON knowledge_collections FOR SELECT USING (true);
CREATE POLICY "knowledge_collections_insert" ON knowledge_collections FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "knowledge_collections_update" ON knowledge_collections FOR UPDATE USING (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "knowledge_collections_delete" ON knowledge_collections FOR DELETE USING (auth.role() IN ('anon', 'service_role'));

-- Knowledge documents policies
CREATE POLICY "knowledge_documents_select" ON knowledge_documents FOR SELECT USING (true);
CREATE POLICY "knowledge_documents_insert" ON knowledge_documents FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "knowledge_documents_update" ON knowledge_documents FOR UPDATE USING (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "knowledge_documents_delete" ON knowledge_documents FOR DELETE USING (auth.role() IN ('anon', 'service_role'));

-- Training datasets policies
CREATE POLICY "training_datasets_select" ON training_datasets FOR SELECT USING (true);
CREATE POLICY "training_datasets_insert" ON training_datasets FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "training_datasets_update" ON training_datasets FOR UPDATE USING (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "training_datasets_delete" ON training_datasets FOR DELETE USING (auth.role() IN ('anon', 'service_role'));

-- Training tasks policies
CREATE POLICY "training_tasks_select" ON training_tasks FOR SELECT USING (true);
CREATE POLICY "training_tasks_insert" ON training_tasks FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "training_tasks_update" ON training_tasks FOR UPDATE USING (auth.role() IN ('anon', 'service_role'));

-- Model versions policies
CREATE POLICY "model_versions_select" ON model_versions FOR SELECT USING (true);
CREATE POLICY "model_versions_insert" ON model_versions FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "model_versions_update" ON model_versions FOR UPDATE USING (auth.role() IN ('anon', 'service_role'));

-- Training metrics policies
CREATE POLICY "training_metrics_select" ON training_metrics FOR SELECT USING (true);
CREATE POLICY "training_metrics_insert" ON training_metrics FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));

-- RAG queries policies
CREATE POLICY "rag_queries_select" ON rag_queries FOR SELECT USING (true);
CREATE POLICY "rag_queries_insert" ON rag_queries FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));

-- System logs policies
CREATE POLICY "system_logs_select" ON system_logs FOR SELECT USING (true);
CREATE POLICY "system_logs_insert" ON system_logs FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));

-- Audit logs policies
CREATE POLICY "audit_logs_select" ON audit_logs FOR SELECT USING (true);
CREATE POLICY "audit_logs_insert" ON audit_logs FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));

-- User sessions policies
CREATE POLICY "user_sessions_select" ON user_sessions FOR SELECT USING (true);
CREATE POLICY "user_sessions_insert" ON user_sessions FOR INSERT WITH CHECK (auth.role() IN ('anon', 'service_role'));
CREATE POLICY "user_sessions_delete" ON user_sessions FOR DELETE USING (auth.role() IN ('anon', 'service_role'));;