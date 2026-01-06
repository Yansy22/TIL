# [Tools] 터미널 명령어와 파이썬 실행 환경 설정

## 1. PowerShell 명령어 주의사항
- **`where` 명령어의 혼선**: PowerShell에서 `where`는 데이터 필터링 도구(`Where-Object`)로 작동한다. 실행 파일 경로를 찾으려면 `where.exe`를 사용하거나 `Get-Command`를 사용해야 한다.
- **가짜 경로(App Alias)**: `...\Microsoft\WindowsApps\python.exe`는 실제 설치 경로가 아닌 바로가기 경로이다. 실제 본체 라이브러리 위치를 찾으려면 파이썬 내부 명령어를 활용해야 한다.

## 2. 주요 터미널 명령어 및 코드

### 가상환경 활성화 및 종료
```powershell
# 1. 가상환경 활성화 (프로젝트 최상위 폴더에서 실행)
.\.venv\Scripts\Activate.ps1

# 2. 가상환경 종료
deactivate
