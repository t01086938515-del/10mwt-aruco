const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  console.log('1. Going to login page...');
  await page.goto('http://localhost:3001/login');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshot_1_login.png' });

  // Login
  console.log('2. Logging in...');
  await page.fill('input[type="text"]', 'admin');
  await page.fill('input[type="password"]', 'admin123');
  await page.click('button[type="submit"]');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'screenshot_2_dashboard.png' });

  // Go to test setup
  console.log('3. Going to test setup...');
  await page.goto('http://localhost:3001/test/setup');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshot_3_setup.png' });

  // Go to upload
  console.log('4. Going to upload...');
  await page.goto('http://localhost:3001/test/ai-upload');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshot_4_upload.png' });

  // Upload video file
  console.log('5. Looking for video file...');
  const fs = require('fs');
  const videoDir = 'C:\\MY APP\\10mwt-integrated\\backend\\uploads';
  let videoFile = null;

  if (fs.existsSync(videoDir)) {
    const files = fs.readdirSync(videoDir);
    const videoFiles = files.filter(f => f.endsWith('.mp4') || f.endsWith('.mov') || f.endsWith('.avi'));
    if (videoFiles.length > 0) {
      videoFile = path.join(videoDir, videoFiles[0]);
      console.log('Found video:', videoFile);
    }
  }

  if (videoFile) {
    // Upload the video
    const fileInput = await page.$('input[type="file"]');
    if (fileInput) {
      await fileInput.setInputFiles(videoFile);
      console.log('6. Video uploaded, waiting...');
      await page.waitForTimeout(3000);
      await page.screenshot({ path: 'screenshot_5_after_upload.png' });
    }
  } else {
    console.log('No video file found in uploads directory');
  }

  // Check if we're on analyze page
  console.log('7. Checking analyze page...');
  await page.goto('http://localhost:3001/test/ai-analyze');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'screenshot_6_analyze.png' });

  // Wait and take more screenshots during analysis
  for (let i = 0; i < 5; i++) {
    await page.waitForTimeout(5000);
    await page.screenshot({ path: `screenshot_7_analyze_${i}.png` });
    console.log(`Screenshot ${i + 1}/5 taken`);
  }

  console.log('Done! Check screenshots in current directory');
  await browser.close();
})();
