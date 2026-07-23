//! webhook HMAC-SHA256署名検証。
//!
//! 仕様（tsuitate-sample-bot README「HMAC検証手順」）:
//! `timestamp + "." + rawBody` に対する HMAC-SHA256 を `sha256={hex}` 形式で
//! `X-Tsuitate-Signature` と比較する。タイムスタンプが許容秒数以上ずれていれば拒否。

use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

pub fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// 署名を検証する。timestamp・signatureは生ヘッダ値をそのまま渡す。
pub fn verify(
    secret: &[u8],
    timestamp: &str,
    signature: &str,
    raw_body: &[u8],
    tolerance_secs: i64,
    now_unix: i64,
) -> bool {
    let Ok(request_time) = timestamp.parse::<i64>() else {
        return false;
    };
    if (now_unix - request_time).abs() > tolerance_secs {
        return false;
    }
    let Some(sig_hex) = signature.strip_prefix("sha256=") else {
        return false;
    };
    let Some(sig_bytes) = decode_hex(sig_hex) else {
        return false;
    };
    let Ok(mut mac) = HmacSha256::new_from_slice(secret) else {
        return false;
    };
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(raw_body);
    mac.verify_slice(&sig_bytes).is_ok()
}

fn decode_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sign(secret: &[u8], timestamp: &str, body: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret).unwrap();
        mac.update(timestamp.as_bytes());
        mac.update(b".");
        mac.update(body);
        let digest = mac.finalize().into_bytes();
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        format!("sha256={hex}")
    }

    #[test]
    fn accepts_correctly_signed_request() {
        let secret = b"topsecret";
        let body = br#"{"type":"your_turn"}"#;
        let ts = "1000";
        let sig = sign(secret, ts, body);
        assert!(verify(secret, ts, &sig, body, 300, 1000));
        // 許容範囲内のずれ
        assert!(verify(secret, ts, &sig, body, 300, 1200));
    }

    #[test]
    fn rejects_wrong_secret() {
        let body = br#"{"type":"your_turn"}"#;
        let ts = "1000";
        let sig = sign(b"topsecret", ts, body);
        assert!(!verify(b"wrongsecret", ts, &sig, body, 300, 1000));
    }

    #[test]
    fn rejects_tampered_body() {
        let secret = b"topsecret";
        let body = br#"{"type":"your_turn"}"#;
        let ts = "1000";
        let sig = sign(secret, ts, body);
        assert!(!verify(secret, ts, &sig, b"{}", 300, 1000));
    }

    #[test]
    fn rejects_stale_timestamp() {
        let secret = b"topsecret";
        let body = br#"{"type":"your_turn"}"#;
        let ts = "1000";
        let sig = sign(secret, ts, body);
        assert!(!verify(secret, ts, &sig, body, 300, 1301));
    }

    #[test]
    fn rejects_malformed_signature() {
        let secret = b"topsecret";
        let body = br#"{}"#;
        assert!(!verify(secret, "1000", "not-hex", body, 300, 1000));
        assert!(!verify(secret, "1000", "sha256=zz", body, 300, 1000));
        assert!(!verify(
            secret,
            "not-a-number",
            "sha256=00",
            body,
            300,
            1000
        ));
    }
}
