// app.js
document.addEventListener('DOMContentLoaded', () => {
    // 音频处理核心对象
    let mediaRecorder;
    let audioContext;
    let analyser;
    
    // 状态变量
    let audioChunks = [];
    let animationFrameId;
    let startTime;
    let totalPausedDuration = 0;
    let pauseStartTime = 0;

    // DOM元素引用
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const timeDisplay = document.getElementById('timeDisplay');
    const statusElement = document.getElementById('status');
    const canvas = document.getElementById('audioWave');
    const canvasCtx = canvas.getContext('2d');

    // 初始化音频处理
    const initAudio = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            initAudioContext(stream);
            initMediaRecorder(stream);
        } catch (error) {
            statusElement.textContent = '麦克风访问被拒绝';
            console.error('音频初始化错误:', error);
        }
    };

    const initAudioContext = (stream) => {
        audioContext = new AudioContext();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize = 256;
    };

    const initMediaRecorder = (stream) => {
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            statusElement.textContent = '上传中...';
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await uploadAudio(audioBlob);
            resetState();
        };
    };

    // 波形绘制
    const drawWaveform = () => {
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = '#ffffff';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = '#ff6b81';
        canvasCtx.beginPath();

        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const amplitude = dataArray[i] / 128.0;
            const y = amplitude * canvas.height / 2;

            i === 0 ? canvasCtx.moveTo(x, y) : canvasCtx.lineTo(x, y);
            x += sliceWidth;
        }

        canvasCtx.stroke();
        animationFrameId = requestAnimationFrame(drawWaveform);
    };

    // 时间管理
    const updateTimer = () => {
        const elapsed = Date.now() - startTime - totalPausedDuration;
        const totalSeconds = Math.floor(elapsed / 1000);
        const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
        const seconds = String(totalSeconds % 60).padStart(2, '0');
        timeDisplay.textContent = `${minutes}:${seconds}`;
    };

    // 录音控制
    const startRecording = () => {
        mediaRecorder.start();
        startTime = Date.now();
        totalPausedDuration = 0;
        
        recordButton.classList.add('recording');
        stopButton.style.display = 'block';
        statusElement.textContent = '录音中...';
        
        drawWaveform();
        timerInterval = setInterval(updateTimer, 200);
    };

    const pauseRecording = () => {
        mediaRecorder.pause();
        pauseStartTime = Date.now();
        
        recordButton.classList.replace('recording', 'paused');
        statusElement.textContent = '已暂停';
        
        clearInterval(timerInterval);
        cancelAnimationFrame(animationFrameId);
    };

    const resumeRecording = () => {
        mediaRecorder.resume();
        totalPausedDuration += Date.now() - pauseStartTime;
        
        recordButton.classList.replace('paused', 'recording');
        statusElement.textContent = '录音中...';
        
        startTime += Date.now() - pauseStartTime;
        timerInterval = setInterval(updateTimer, 200);
        drawWaveform();
    };

    const stopRecording = () => {
        mediaRecorder.stop();
        stopButton.style.display = 'none';
        recordButton.classList.remove('recording', 'paused');
    };

    // 文件上传
    const uploadAudio = async (blob) => {
        const formData = new FormData();
        formData.append('audio', blob, `recording_${Date.now()}.wav`);

        try {
            const response = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                body: formData
            });

            response.ok ? 
                statusElement.textContent = '上传成功!' :
                statusElement.textContent = '上传失败';
        } catch (error) {
            statusElement.textContent = '网络错误';
            console.error('上传错误:', error);
        }
    };

    // 状态重置
    const resetState = () => {
        audioChunks = [];
        totalPausedDuration = 0;
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        timeDisplay.textContent = '00:00';
    };

    // 事件监听
    recordButton.addEventListener('click', () => {
        if (!mediaRecorder) return;

        switch(mediaRecorder.state) {
            case 'inactive': 
                startRecording();
                break;
            case 'recording': 
                pauseRecording();
                break;
            case 'paused': 
                resumeRecording();
                break;
        }
    });

    stopButton.addEventListener('click', stopRecording);

    // 启动初始化
    initAudio();

    // 窗口调整处理
    window.addEventListener('resize', () => {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    });
});