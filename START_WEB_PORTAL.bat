@echo off
chcp 65001 > nul
echo ========================================
echo VeriSafe Web Portal 시작
echo ========================================
echo.

cd web-portal

echo [1/2] 패키지 확인 중...
if not exist "node_modules\" (
    echo node_modules가 없습니다. npm install을 실행합니다...
    call npm install
    if errorlevel 1 (
        echo.
        echo ❌ npm install 실패!
        pause
        exit /b 1
    )
) else (
    echo ✓ node_modules 존재
)

echo.
echo [2/2] 웹 포털 개발 서버 시작...
echo.
echo 📌 웹 포털이 http://localhost:3000 에서 실행됩니다
echo 📌 매퍼/관리자 계정으로 로그인하세요
echo.
echo Ctrl+C를 눌러 서버를 종료할 수 있습니다.
echo.

call npm run dev

pause
