import pandas as pd
import requests
from pathlib import Path

API_URL = "https://finance.naver.com/api/sise/etfItemList.naver?etfType=0"
# etfType:
# 0: 전체, 1: 국내, 2: 해외

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://finance.naver.com/sise/etf.naver",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

def fetch_etf_list():
    resp = requests.get(API_URL, headers = HEADERS, timeout = 15)
    resp.raise_for_status
    data = resp.json()
    items = data.get("result", {}).get("etfItemList", [])
    return pd.json_normalize(items)

if __name__ == "__main__":
    df = fetch_etf_list()
    print(f"[INFO] ETF {len(df)}개 로드")
    
    # 주요 컬럼 rename(가독성)
    rename_map = {
        "itemcode": "종목코드",
        "itemname": "종목명",
        "nowVal": "현재가",
        "risefall": "등락구분",   # 2:상승, 3:보합, 4:하락 (케이스에 따라 다를 수 있음)
        "changeVal": "전일비",
        "changeRate": "등락률(%)",
        "quant": "거래량",
        "amnt": "거래대금",
        "marketSum": "시가총액(억)",
        "nav": "NAV",
        "navChangeRate": "NAV등락률(%)",
        "threeMonthEarnRate": "3개월수익률(%)",
        "quantLastDay": "전일거래량",
        "etfTabCode": "분류코드",
    }
    df = df.rename(columns = rename_map)
    
    # 저장
    out_dir = Path("data")
    out_dir.mkdir(exist_ok = True)
    out_path = out_dir / "etf_list.csv"
    df.to_csv(out_path, index = False, encoding = "utf-8-sig")
    print(f"[DONE] 저장 완료: {out_path.resolve()}")