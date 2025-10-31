class VADProcessor extends AudioWorkletProcessor {
constructor(){
    super();
    // 적응형 노이즈바닥
    this.noise = 0.0;
    this.alpha = 0.95; // 노이즈 업데이트 속도
    this.margin = 0.008; // 말로 판단할 여유 마진
    this.hpf_y = 0.0; // 간단한 1차 하이패스용
    this.hpf_prev = 0.0;
}
static get parameterDescriptors(){ return []; }


process(inputs, outputs, parameters){
    const input = inputs && inputs[0] && inputs[0][0];
    if (!input) return true;


    // 간단한 1차 하이패스 + RMS
    let sum=0.0;
    let zc = 0;
    let prev = 0;
    const a = 0.995; // ~100Hz 근처의 간단 HPF 느낌 (SR 16k 기준 대략)
    for (let i=0;i<input.length;i++){
        const x = input[i];
        const y = x - this.hpf_prev + a * this.hpf_y;
        this.hpf_prev = x;
        this.hpf_y = y;


        sum += y*y;
        if (i>0 && ((y>=0)!=(prev>=0))) zc++;
        prev = y;
    }
    const rms = Math.sqrt(sum / input.length);


    // 발화 중 업데이트 억제는 메인 스레드에서 판단하므로 이곳은 항상 추정
    this.noise = this.alpha * this.noise + (1-this.alpha) * rms;
    const voice = rms > (this.noise + this.margin);


    this.port.postMessage({ rms, noise: this.noise, zcr: zc/input.length, voice });
    return true;
}
}
registerProcessor('vad-processor', VADProcessor);