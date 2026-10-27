# Data source record

- Original source file: `C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv`
- Project copy used for local execution: `data/raw/telco.csv`
- Canonical source repository: IBM Telco Customer Churn on ICP4D
- Commit-pinned CSV: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv`
- SHA-256 of the local source file: `62ca518b1ebbcd4c832c18c1a21629b03abe9033fef66fe2e38d8d030658a20c`

To verify the byte identity locally:

```powershell
$py = 'C:\Users\khadi\AppData\Local\Programs\Python\Python312\python.exe'
& $py -c "import hashlib; from pathlib import Path; p = Path(r'C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

This file is treated as the authoritative input snapshot and is left unchanged by the project workflow.
