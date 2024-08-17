### CHANGED FROM ORIGINAL :
- Modified CE model (see [FBankCrossEntropyNetV2](./models/cross_entropy_model.py))
- Modified Linear Adapter for speaker classification (see [DynamicLinearClassifier](./models/classifier.py))

### TODO :
- [] Data preprocessing pipeline for raw waveform input
### NOTE :
- Mô hình của Hưng Phạm đang sử dụng có vẻ là mô hình đã được train thêm 1 bước học tương phản. (Will be implement)
- Cấu hình thay đổi trong cả 3 file : thêm số lớp cho mô hình(num_layers)

### RUN :
- Test luồng làm việc chính trong 3 file [authentication.py](./authentication.py) , [classification.py](./classification.py) và [identity.py](./identity.py)
- Cả 3 file này, 3 hàm train,test và infer có thể test bằng cách chuyển async def, thêm cấu hình -> def, đổi hàm trong main và run file
- Check các sample mẫu