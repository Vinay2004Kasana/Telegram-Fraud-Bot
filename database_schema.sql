-- Enhanced Database Schema for Financial Fraud Detection Bot
-- Execute this in your Supabase SQL Editor

-- Enable UUID extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table for storing suspicious keywords with categories and weights
CREATE TABLE IF NOT EXISTS suspicious_keywords (
    id BIGSERIAL PRIMARY KEY,
    keyword VARCHAR(255) NOT NULL UNIQUE,
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    weight INTEGER NOT NULL DEFAULT 1,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for comprehensive fraud analyses
CREATE TABLE IF NOT EXISTS fraud_analyses (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical', 'error')),
    fraud_score INTEGER DEFAULT 0,
    analysis_details JSONB NOT NULL DEFAULT '{}',
    mentioned_leaders TEXT[] DEFAULT '{}',
    mentioned_stocks TEXT[] DEFAULT '{}',
    mentioned_agencies TEXT[] DEFAULT '{}',
    file_hash VARCHAR(32),
    content_type VARCHAR(20), -- 'text', 'image', 'video', 'audio', 'mixed'
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for high-priority fraud alerts
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    analysis_id BIGINT REFERENCES fraud_analyses(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL DEFAULT 'financial_manipulation',
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('medium', 'high', 'critical')),
    title VARCHAR(255),
    description TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'false_positive')),
    assigned_to VARCHAR(255),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for tracking media file hashes (duplicate detection)
CREATE TABLE IF NOT EXISTS media_fingerprints (
    id BIGSERIAL PRIMARY KEY,
    file_hash VARCHAR(32) NOT NULL UNIQUE,
    media_type VARCHAR(20) NOT NULL CHECK (media_type IN ('image', 'video', 'audio', 'document')),
    file_size_bytes INTEGER,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    times_seen INTEGER DEFAULT 1,
    is_flagged BOOLEAN DEFAULT FALSE,
    flagged_reason TEXT,
    analysis_results JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for user profiles and reputation tracking
CREATE TABLE IF NOT EXISTS user_profiles (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE,
    username VARCHAR(255),
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    total_analyses INTEGER DEFAULT 0,
    high_risk_content_count INTEGER DEFAULT 0,
    false_positive_reports INTEGER DEFAULT 0,
    reputation_score DECIMAL(3,2) DEFAULT 0.50 CHECK (reputation_score >= 0.0 AND reputation_score <= 1.0),
    risk_category VARCHAR(20) DEFAULT 'unknown' CHECK (risk_category IN ('low', 'medium', 'high', 'critical', 'unknown')),
    is_verified BOOLEAN DEFAULT FALSE,
    is_blocked BOOLEAN DEFAULT FALSE,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for storing corporate leaders information
CREATE TABLE IF NOT EXISTS corporate_leaders (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL, -- lowercase, standardized
    position VARCHAR(255),
    company VARCHAR(255),
    company_symbols TEXT[] DEFAULT '{}',
    industry VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    importance_score INTEGER DEFAULT 1, -- 1-10 scale
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(normalized_name)
);

-- Table for monitored stock symbols
CREATE TABLE IF NOT EXISTS monitored_stocks (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    exchange VARCHAR(10),
    sector VARCHAR(100),
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT TRUE,
    monitoring_level VARCHAR(10) DEFAULT 'normal' CHECK (monitoring_level IN ('low', 'normal', 'high', 'critical')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for regulatory agencies
CREATE TABLE IF NOT EXISTS regulatory_agencies (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    abbreviation VARCHAR(20),
    country VARCHAR(3),
    agency_type VARCHAR(50), -- 'securities', 'banking', 'insurance', etc.
    official_website VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    authority_level INTEGER DEFAULT 1, -- 1-10 scale
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for bot performance metrics
CREATE TABLE IF NOT EXISTS bot_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    metric_type VARCHAR(50), -- 'counter', 'gauge', 'histogram'
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for system logs and audit trail
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL,
    event_type VARCHAR(50),
    user_id BIGINT,
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default suspicious keywords
INSERT INTO suspicious_keywords (keyword, category, weight, description) VALUES
-- Urgency and manipulation terms
('breaking news', 'urgency', 3, 'Creates false sense of urgency'),
('urgent', 'urgency', 2, 'Urgency manipulation tactic'),
('emergency', 'urgency', 3, 'Emergency false urgency'),
('immediate action', 'urgency', 3, 'Immediate action manipulation'),
('act now', 'urgency', 2, 'Act now manipulation'),
('limited time', 'urgency', 2, 'Limited time pressure tactic'),
('exclusive', 'urgency', 2, 'False exclusivity claim'),

-- Financial fraud terms
('sec investigation', 'regulatory', 5, 'SEC investigation claims'),
('finra fine', 'regulatory', 4, 'FINRA penalty claims'),
('insider trading', 'fraud', 5, 'Insider trading allegations'),
('market manipulation', 'fraud', 5, 'Market manipulation claims'),
('pump and dump', 'fraud', 5, 'Pump and dump schemes'),
('ponzi scheme', 'fraud', 4, 'Ponzi scheme references'),
('pyramid scheme', 'fraud', 4, 'Pyramid scheme references'),
('securities fraud', 'fraud', 4, 'Securities fraud allegations'),

-- Deepfake and manipulation
('deepfake', 'manipulation', 5, 'Deepfake content references'),
('deep fake', 'manipulation', 5, 'Deep fake content references'),
('ai generated', 'manipulation', 4, 'AI generated content'),
('synthetic media', 'manipulation', 4, 'Synthetic media content'),
('fake video', 'manipulation', 4, 'Fake video claims'),
('voice clone', 'manipulation', 4, 'Voice cloning references'),

-- Regulatory terms
('cease and desist', 'regulatory', 4, 'Regulatory cease and desist orders'),
('regulatory approval', 'regulatory', 3, 'Regulatory approval claims'),
('compliance violation', 'regulatory', 3, 'Compliance violation claims'),
('enforcement action', 'regulatory', 4, 'Regulatory enforcement claims'),
('emergency order', 'regulatory', 4, 'Emergency regulatory orders'),

-- Financial impact terms
('bankruptcy', 'financial', 4, 'Bankruptcy announcements'),
('merger', 'financial', 3, 'Merger announcements'),
('acquisition', 'financial', 3, 'Acquisition announcements'),
('earnings surprise', 'financial', 2, 'Earnings surprise claims'),
('stock split', 'financial', 2, 'Stock split announcements'),
('dividend increase', 'financial', 2, 'Dividend increase claims'),

-- Leak and insider terms
('leaked', 'insider', 3, 'Leaked information claims'),
('confidential', 'insider', 3, 'Confidential information claims'),
('insider information', 'insider', 4, 'Insider information claims'),
('whistleblower', 'insider', 2, 'Whistleblower claims'),
('exclusive leak', 'insider', 3, 'Exclusive leak claims')

ON CONFLICT (keyword) DO NOTHING;

-- Insert corporate leaders
INSERT INTO corporate_leaders (name, normalized_name, position, company, company_symbols, industry, importance_score) VALUES
('Elon Musk', 'elon musk', 'CEO', 'Tesla Inc.', '{"TSLA"}', 'Automotive/Technology', 10),
('Tim Cook', 'tim cook', 'CEO', 'Apple Inc.', '{"AAPL"}', 'Technology', 9),
('Satya Nadella', 'satya nadella', 'CEO', 'Microsoft Corporation', '{"MSFT"}', 'Technology', 9),
('Jeff Bezos', 'jeff bezos', 'Executive Chairman', 'Amazon.com Inc.', '{"AMZN"}', 'E-commerce', 9),
('Mark Zuckerberg', 'mark zuckerberg', 'CEO', 'Meta Platforms Inc.', '{"META"}', 'Technology', 8),
('Sundar Pichai', 'sundar pichai', 'CEO', 'Alphabet Inc.', '{"GOOGL","GOOG"}', 'Technology', 8),
('Jensen Huang', 'jensen huang', 'CEO', 'NVIDIA Corporation', '{"NVDA"}', 'Technology', 8),
('Jamie Dimon', 'jamie dimon', 'CEO', 'JPMorgan Chase & Co.', '{"JPM"}', 'Financial Services', 7),
('Warren Buffett', 'warren buffett', 'CEO', 'Berkshire Hathaway Inc.', '{"BRK-A","BRK-B"}', 'Conglomerate', 9),
('Mary Barra', 'mary barra', 'CEO', 'General Motors Company', '{"GM"}', 'Automotive', 6),
('Andy Jassy', 'andy jassy', 'CEO', 'Amazon.com Inc.', '{"AMZN"}', 'E-commerce', 7),
('Lisa Su', 'lisa su', 'CEO', 'Advanced Micro Devices Inc.', '{"AMD"}', 'Technology', 6),
('Reed Hastings', 'reed hastings', 'Co-CEO', 'Netflix Inc.', '{"NFLX"}', 'Entertainment', 6),
('Brian Chesky', 'brian chesky', 'CEO', 'Airbnb Inc.', '{"ABNB"}', 'Travel/Technology', 5),
('Daniel Ek', 'daniel ek', 'CEO', 'Spotify Technology S.A.', '{"SPOT"}', 'Entertainment/Technology', 4)

ON CONFLICT (normalized_name) DO NOTHING;

-- Insert monitored stocks
INSERT INTO monitored_stocks (symbol, company_name, exchange, sector, monitoring_level) VALUES
('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology', 'high'),
('GOOGL', 'Alphabet Inc. Class A', 'NASDAQ', 'Technology', 'high'),
('GOOG', 'Alphabet Inc. Class C', 'NASDAQ', 'Technology', 'high'),
('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology', 'high'),
('TSLA', 'Tesla Inc.', 'NASDAQ', 'Automotive', 'critical'),
('AMZN', 'Amazon.com Inc.', 'NASDAQ', 'E-commerce', 'high'),
('META', 'Meta Platforms Inc.', 'NASDAQ', 'Technology', 'high'),
('NVDA', 'NVIDIA Corporation', 'NASDAQ', 'Technology', 'high'),
('JPM', 'JPMorgan Chase & Co.', 'NYSE', 'Financial Services', 'high'),
('BRK-A', 'Berkshire Hathaway Inc. Class A', 'NYSE', 'Conglomerate', 'high'),
('BRK-B', 'Berkshire Hathaway Inc. Class B', 'NYSE', 'Conglomerate', 'high'),
('NFLX', 'Netflix Inc.', 'NASDAQ', 'Entertainment', 'normal'),
('AMD', 'Advanced Micro Devices Inc.', 'NASDAQ', 'Technology', 'normal'),
('GM', 'General Motors Company', 'NYSE', 'Automotive', 'normal'),
('ABNB', 'Airbnb Inc.', 'NASDAQ', 'Travel', 'normal'),
('SPOT', 'Spotify Technology S.A.', 'NYSE', 'Entertainment', 'normal')

ON CONFLICT (symbol) DO NOTHING;

-- Insert regulatory agencies
INSERT INTO regulatory_agencies (name, abbreviation, country, agency_type, official_website, authority_level) VALUES
('Securities and Exchange Commission', 'SEC', 'USA', 'securities', 'https://www.sec.gov', 10),
('Financial Industry Regulatory Authority', 'FINRA', 'USA', 'securities', 'https://www.finra.org', 8),
('Commodity Futures Trading Commission', 'CFTC', 'USA', 'commodities', 'https://www.cftc.gov', 7),
('Federal Reserve System', 'FED', 'USA', 'banking', 'https://www.federalreserve.gov', 10),
('Federal Deposit Insurance Corporation', 'FDIC', 'USA', 'banking', 'https://www.fdic.gov', 8),
('Office of the Comptroller of the Currency', 'OCC', 'USA', 'banking', 'https://www.occ.gov', 7),
('Securities and Exchange Board of India', 'SEBI', 'IND', 'securities', 'https://www.sebi.gov.in', 9),
('Financial Conduct Authority', 'FCA', 'GBR', 'financial', 'https://www.fca.org.uk', 9),
('Reserve Bank of India', 'RBI', 'IND', 'banking', 'https://www.rbi.org.in', 9),
('Bank of England', 'BOE', 'GBR', 'banking', 'https://www.bankofengland.co.uk', 9)

ON CONFLICT DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_fraud_analyses_user_id ON fraud_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_fraud_analyses_timestamp ON fraud_analyses(analysis_timestamp);
CREATE INDEX IF NOT EXISTS idx_fraud_analyses_risk_level ON fraud_analyses(risk_level);
CREATE INDEX IF NOT EXISTS idx_fraud_analyses_fraud_score ON fraud_analyses(fraud_score);
CREATE INDEX IF NOT EXISTS idx_fraud_analyses_file_hash ON fraud_analyses(file_hash);

CREATE INDEX IF NOT EXISTS idx_fraud_alerts_severity ON fraud_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status ON fraud_alerts(status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_timestamp ON fraud_alerts(timestamp);

CREATE INDEX IF NOT EXISTS idx_media_fingerprints_hash ON media_fingerprints(file_hash);
CREATE INDEX IF NOT EXISTS idx_media_fingerprints_flagged ON media_fingerprints(is_flagged);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_reputation ON user_profiles(reputation_score);

CREATE INDEX IF NOT EXISTS idx_suspicious_keywords_category ON suspicious_keywords(category);
CREATE INDEX IF NOT EXISTS idx_suspicious_keywords_active ON suspicious_keywords(is_active);

CREATE INDEX IF NOT EXISTS idx_corporate_leaders_normalized ON corporate_leaders(normalized_name);
CREATE INDEX IF NOT EXISTS idx_monitored_stocks_symbol ON monitored_stocks(symbol);

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_fraud_analyses_details ON fraud_analyses USING GIN (analysis_details);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_details ON fraud_alerts USING GIN (details);

-- Create functions for common operations

-- Function to update user reputation
CREATE OR REPLACE FUNCTION update_user_reputation(p_user_id BIGINT, p_fraud_score INTEGER, p_is_false_positive BOOLEAN DEFAULT FALSE)
RETURNS VOID AS $$
DECLARE
    current_reputation DECIMAL(3,2);
    reputation_change DECIMAL(3,2);
    new_reputation DECIMAL(3,2);
BEGIN
    -- Get or create user profile
    INSERT INTO user_profiles (user_id, total_analyses)
    VALUES (p_user_id, 0)
    ON CONFLICT (user_id) DO NOTHING;
    
    SELECT reputation_score INTO current_reputation
    FROM user_profiles
    WHERE user_id = p_user_id;
    
    -- Calculate reputation change
    IF p_is_false_positive THEN
        reputation_change := 0.05; -- Slight increase for false positive reports
    ELSIF p_fraud_score >= 20 THEN
        reputation_change := -0.15; -- Large decrease for critical fraud
    ELSIF p_fraud_score >= 15 THEN
        reputation_change := -0.10; -- Medium decrease for high fraud
    ELSIF p_fraud_score >= 8 THEN
        reputation_change := -0.05; -- Small decrease for medium fraud
    ELSE
        reputation_change := 0.01; -- Small increase for clean content
    END IF;
    
    -- Apply change and clamp to valid range
    new_reputation := GREATEST(0.0, LEAST(1.0, current_reputation + reputation_change));
    
    -- Update user profile
    UPDATE user_profiles
    SET reputation_score = new_reputation,
        total_analyses = total_analyses + 1,
        high_risk_content_count = CASE 
            WHEN p_fraud_score >= 15 THEN high_risk_content_count + 1 
            ELSE high_risk_content_count 
        END,
        false_positive_reports = CASE 
            WHEN p_is_false_positive THEN false_positive_reports + 1 
            ELSE false_positive_reports 
        END,
        risk_category = CASE 
            WHEN new_reputation < 0.2 THEN 'critical'
            WHEN new_reputation < 0.4 THEN 'high'
            WHEN new_reputation < 0.6 THEN 'medium'
            ELSE 'low'
        END,
        last_activity = NOW(),
        updated_at = NOW()
    WHERE user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check and update media fingerprints
CREATE OR REPLACE FUNCTION check_media_duplicate(p_file_hash VARCHAR(32), p_media_type VARCHAR(20), p_file_size INTEGER DEFAULT NULL)
RETURNS TABLE(is_duplicate BOOLEAN, times_seen INTEGER, is_flagged BOOLEAN) AS $$
DECLARE
    existing_record RECORD;
BEGIN
    SELECT * INTO existing_record 
    FROM media_fingerprints 
    WHERE file_hash = p_file_hash;
    
    IF FOUND THEN
        -- Update existing record
        UPDATE media_fingerprints 
        SET times_seen = times_seen + 1,
            updated_at = NOW()
        WHERE file_hash = p_file_hash;
        
        RETURN QUERY SELECT TRUE, existing_record.times_seen + 1, existing_record.is_flagged;
    ELSE
        -- Insert new record
        INSERT INTO media_fingerprints (file_hash, media_type, file_size_bytes, times_seen)
        VALUES (p_file_hash, p_media_type, p_file_size, 1);
        
        RETURN QUERY SELECT FALSE, 1, FALSE;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to get fraud statistics
CREATE OR REPLACE FUNCTION get_fraud_statistics(p_days INTEGER DEFAULT 7)
RETURNS TABLE(
    total_analyses BIGINT,
    high_risk_count BIGINT,
    critical_risk_count BIGINT,
    avg_fraud_score NUMERIC,
    unique_users BIGINT,
    deepfake_detections BIGINT,
    document_fraud_detections BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_analyses,
        COUNT(*) FILTER (WHERE risk_level = 'high') as high_risk_count,
        COUNT(*) FILTER (WHERE risk_level = 'critical') as critical_risk_count,
        ROUND(AVG(fraud_score), 2) as avg_fraud_score,
        COUNT(DISTINCT user_id) as unique_users,
        COUNT(*) FILTER (WHERE analysis_details ? 'deepfake_probability') as deepfake_detections,
        COUNT(*) FILTER (WHERE analysis_details ? 'is_document') as document_fraud_detections
    FROM fraud_analyses
    WHERE analysis_timestamp >= NOW() - INTERVAL '1 day' * p_days;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for dashboard statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS fraud_statistics_daily AS
SELECT 
    DATE(analysis_timestamp) as analysis_date,
    risk_level,
    COUNT(*) as count,
    AVG(fraud_score) as avg_fraud_score,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(*) FILTER (WHERE mentioned_leaders != '{}') as leader_mentions,
    COUNT(*) FILTER (WHERE mentioned_stocks != '{}') as stock_mentions
FROM fraud_analyses 
WHERE analysis_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(analysis_timestamp), risk_level;

CREATE INDEX IF NOT EXISTS idx_fraud_statistics_daily_date ON fraud_statistics_daily(analysis_date);

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_fraud_statistics()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW fraud_statistics_daily;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update user reputation after analysis
CREATE OR REPLACE FUNCTION trigger_update_user_reputation()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM update_user_reputation(NEW.user_id, NEW.fraud_score, FALSE);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_update_user_reputation
    AFTER INSERT ON fraud_analyses
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_user_reputation();

-- Create row-level security policies (optional)
ALTER TABLE fraud_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE fraud_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Allow service role to access all data
CREATE POLICY "Service role can access all fraud_analyses" ON fraud_analyses
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can access all fraud_alerts" ON fraud_alerts
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can access all user_profiles" ON user_profiles
    FOR ALL USING (auth.role() = 'service_role');

-- Comments for documentation
COMMENT ON TABLE fraud_analyses IS 'Comprehensive fraud analysis results for all content types';
COMMENT ON TABLE fraud_alerts IS 'High-priority alerts for suspected financial fraud and market manipulation';
COMMENT ON TABLE media_fingerprints IS 'File hashes for duplicate detection and viral content tracking';
COMMENT ON TABLE user_profiles IS 'User behavior tracking and reputation scoring';
COMMENT ON TABLE corporate_leaders IS 'Database of corporate leaders for deepfake detection';
COMMENT ON TABLE monitored_stocks IS 'Stock symbols actively monitored for manipulation';
COMMENT ON TABLE regulatory_agencies IS 'Regulatory agencies and their authority levels';
COMMENT ON TABLE suspicious_keywords IS 'Dynamic keyword database for fraud detection';

-- Final setup message
DO $$
BEGIN
    RAISE NOTICE 'Financial Fraud Detection database schema setup completed successfully!';
    RAISE NOTICE 'Tables created: %, %, %, %, %, %, %, %', 
        'fraud_analyses', 'fraud_alerts', 'media_fingerprints', 'user_profiles',
        'corporate_leaders', 'monitored_stocks', 'regulatory_agencies', 'suspicious_keywords';
    RAISE NOTICE 'Sample data inserted for keywords, leaders, stocks, and agencies';
    RAISE NOTICE 'Indexes, functions, and triggers configured';
    RAISE NOTICE 'Your fraud detection bot is ready to use!';
END $$;