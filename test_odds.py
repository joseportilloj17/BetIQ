import requests
ODDSPAPI = 'a1c32a97-45a3-4e1a-96cf-07d0da4588fc'
BASE = 'https://api.oddspapi.io/v4'
r = requests.get(f'{BASE}/fixtures', params={'apiKey': ODDSPAPI, 'sportId': 11, 'from': '2024-01-15T00:00:00Z', 'to': '2024-01-16T00:00:00Z'}, timeout=15)
data = r.json()
print(f'Fixtures: {r.status_code} - {len(data)} games')
if data:
    fix = data[0]
    fid = fix.get('fixtureId')
    print(str(fix.get('participant1Name')) + ' vs ' + str(fix.get('participant2Name')))
    h = requests.get(f'{BASE}/historical-odds', params={'apiKey': ODDSPAPI, 'fixtureId': fid}, timeout=15)
    print(f'Historical: {h.status_code} - ' + h.text[:200])
