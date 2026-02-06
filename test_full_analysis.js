const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const browser = await chromium.launch({ headless: false, slowMo: 500 });
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    // 1. Login
    console.log('1. Logging in...');
    await page.goto('http://localhost:3001/login');
    await page.waitForTimeout(1000);

    // 데모 계정으로 시작하기 버튼 클릭
    await page.click('text=데모 계정으로 시작하기');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: 'test_1_after_login.png' });
    console.log('Login successful');

    // 2. Go to test setup
    console.log('2. Going to test setup...');
    await page.click('text=검사 시작');
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'test_2_setup.png' });

    // 3. Select patient and continue
    console.log('3. Selecting patient...');
    // 첫 번째 환자 선택
    const patientCard = await page.$('.cursor-pointer');
    if (patientCard) {
      await patientCard.click();
      await page.waitForTimeout(500);
    }

    // 다음 버튼 클릭
    const nextBtn = await page.$('button:has-text("다음")');
    if (nextBtn) {
      await nextBtn.click();
      await page.waitForTimeout(1000);
    }
    await page.screenshot({ path: 'test_3_after_patient.png' });

    // 4. Upload video
    console.log('4. Uploading video...');
    await page.waitForTimeout(1000);

    const videoFile = 'C:\\MY APP\\10mwtapp\\7.11_7.56.mp4';
    const fileInput = await page.$('input[type="file"]');
    if (fileInput) {
      await fileInput.setInputFiles(videoFile);
      console.log('Video file selected');
      await page.waitForTimeout(3000);
    }
    await page.screenshot({ path: 'test_4_after_upload.png' });

    // 5. Wait for analysis page
    console.log('5. Waiting for analysis...');
    await page.waitForURL('**/ai-analyze**', { timeout: 30000 }).catch(() => {
      console.log('Not redirected to analyze page, checking current URL');
    });

    console.log('Current URL:', page.url());
    await page.screenshot({ path: 'test_5_analyze_start.png' });

    // 6. Take screenshots during analysis
    console.log('6. Taking screenshots during analysis...');
    for (let i = 0; i < 10; i++) {
      await page.waitForTimeout(5000);
      await page.screenshot({ path: `test_6_analyze_${i}.png` });
      console.log(`Analysis screenshot ${i + 1}/10`);

      // Check if we're on result page
      if (page.url().includes('ai-result')) {
        console.log('Analysis complete! On result page');
        break;
      }
    }

    // 7. Final screenshot
    await page.screenshot({ path: 'test_7_final.png' });
    console.log('Done! Screenshots saved');

  } catch (error) {
    console.error('Error:', error.message);
    await page.screenshot({ path: 'test_error.png' });
  }

  // Keep browser open for inspection
  console.log('Browser will stay open for 60 seconds...');
  await page.waitForTimeout(60000);
  await browser.close();
})();
