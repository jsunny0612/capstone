#!/usr/bin/env python

import time

import torch
import torchvision.utils
from fvcore.common.checkpoint import Checkpointer

from gaze_estimation import (
    create_dataloader,
    create_logger,
    create_loss,
    create_optimizer,
    create_scheduler,
    create_tensorboard_writer,
)
from gaze_estimation import GazeEstimationMethod, create_model
from gaze_estimation.utils import (
    AverageMeter,
    compute_angle_error,
    create_train_output_dir,
    load_config,
    save_config,
    set_seeds,
    setup_cudnn,
)


def train(epoch, model, optimizer, scheduler, loss_function, train_loader,
          config, tensorboard_writer, logger):
    logger.info(f'Train {epoch}')
# 현재 학습 에포크 로그에 기록
    model.train()

    device = torch.device(config.device)

    loss_meter = AverageMeter() # 손실 값 평균 계산
    angle_error_meter = AverageMeter() # 각도 오차 평균 계산
    start = time.time() # 학습 시간 측정 위해 시작 시간 기록
    for step, (images, poses, gazes) in enumerate(train_loader):
        #train_loader에서 데이터 하나씩 가져와서 반복
        #poses : 포즈 데이터, gazes : 시선 데이터, images : 이미지 데이터
        if config.tensorboard.train_images and step == 0:
            image = torchvision.utils.make_grid(images,
                                                normalize=True,
                                                scale_each=True)
            tensorboard_writer.add_image('Train/Image', image, epoch)
            #학습 과정 첫 번째 스텝에서만 학습 이미지를 Tensorboard에 기록

        images = images.to(device)
        poses = poses.to(device)
        gazes = gazes.to(device)

        optimizer.zero_grad()
        #이전 배치에서 계산된 기울기 초기화

        if config.mode == GazeEstimationMethod.MPIIGaze.name:
            outputs = model(images, poses)
        elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            outputs = model(images)
        else:
            raise ValueError
        # 모델 추정 방식에 따라 MPIIGaze와 MPIIFaceGaze 중 택 1
        loss = loss_function(outputs, gazes)
        # 모델 출력값과 실제 시선값 비교하여 손실 값 계산
        loss.backward()
        # 역전파 수행으로 모델 가중치 기울기 계산
        optimizer.step()
        # 계산된 기울기 이용해서 모델 가중치 업데이트
        angle_error = compute_angle_error(outputs, gazes).mean()
        # 모델 예측값과 실제 시선값 사이의 각도 오차 계산
        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)
        # 배치당 손실 값과 각도 오차 값을 업데이트하여 평균 계산
        if step % config.train.log_period == 0:
            logger.info(f'Epoch {epoch} '
                        f'Step {step}/{len(train_loader)} '
                        f'lr {scheduler.get_last_lr()[0]:.6f} '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'angle error {angle_error_meter.val:.2f} '
                        f'({angle_error_meter.avg:.2f})')
        # 일정 스텝마다 에포크, 손실 값, 각도 오차 등 기록
    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')
    #학습 시간 로그에 기록
    tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    tensorboard_writer.add_scalar('Train/lr',
                                  scheduler.get_last_lr()[0], epoch)
    tensorboard_writer.add_scalar('Train/AngleError', angle_error_meter.avg,
                                  epoch)
    tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)
    #에포크 끝난 후 손실 값, 학습률, 각도 오차, 학습 시간 tensorboard에 기록

def validate(epoch, model, loss_function, val_loader, config,
             tensorboard_writer, logger):
    #model 검증 
    logger.info(f'Val {epoch}')

    model.eval()
    #검증 시작 및 모델 평가 모드 전환
    device = torch.device(config.device)
    
    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    # train 함수와 동일하게 장치 설정


    with torch.no_grad():
    # 검증 중에는 기울기 계산 x
        for step, (images, poses, gazes) in enumerate(val_loader):
        # 검증 데이터 반복
            if config.tensorboard.val_images and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(images,
                                                    normalize=True,
                                                    scale_each=True)
                tensorboard_writer.add_image('Val/Image', image, epoch)
                # 첫 번째 에포크와 첫 번째 스텝에서만 검증 이미지 tensorBoard에 기록

            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)
            # 입력 데이터 장치 전송

            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                outputs = model(images, poses)
            elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                outputs = model(images)
            else:
                raise ValueError
            # 학습과 동일하게 모델 시선 추정 방식 선택
            loss = loss_function(outputs, gazes)

            angle_error = compute_angle_error(outputs, gazes).mean()

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)
            #손실 값과 각도 오차 값 평균으로 업데이트

    logger.info(f'Epoch {epoch} '
                f'loss {loss_meter.avg:.4f} '
                f'angle error {angle_error_meter.avg:.2f}')
    # 검증 결과 로그에 기록

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')
    #검증 소요 시간 로그에 기록

    if epoch > 0:
        tensorboard_writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/AngleError', angle_error_meter.avg,
                                      epoch)
    tensorboard_writer.add_scalar('Val/Time', elapsed, epoch)
    # 검증 결과 tensorboard에 기록

    if config.tensorboard.model_params:
        for name, param in model.named_parameters():
            tensorboard_writer.add_histogram(name, param, epoch)


def main():
    config = load_config()
    #설정 파일 불러오기

    set_seeds(config.train.seed)
    setup_cudnn(config)
    #시드 값 설정하여 재현성 보장 및 cuda 설정 최적화

    output_dir = create_train_output_dir(config)
    save_config(config, output_dir)
    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    #출력 디렉터리 생성, 설정 파일 저장, 로그 기록 객체 생성

    train_loader, val_loader = create_dataloader(config, is_train=True)
    model = create_model(config)
    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    #데이터 로더, 모델, 손실 함수, 최적화 알고리즘, 스케줄러 생성
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir.as_posix(),
                                save_to_disk=True)
    # model : 학습 중인 모델 객체, optimizer: 학습 중인 최적화 알고리즘 상태 저장
    # schedular : 학습률 스케줄러 상태 저장, save_dir : 체크 포인트 저장 경로
    # save_to_disk : True 설정하면 체크포인트를 디스크에 저장
    tensorboard_writer = create_tensorboard_writer(config, output_dir)
    # tesnorboard에 학습 과정 기록을 위해 tensorboard_writer 생성
    if config.train.val_first:
        validate(0, model, loss_function, val_loader, config,
                 tensorboard_writer, logger)
    # 학습 시작 전에 검증 한 번 실행할지 여부 설정
    # val_first 참일 경우 첫 번째 에포크 전에 검증 수행

    for epoch in range(1, config.scheduler.epochs + 1):
        train(epoch, model, optimizer, scheduler, loss_function, train_loader,
              config, tensorboard_writer, logger)
        scheduler.step()
        # 에포크동안 모델 학습 수행
        # 각 에포크 끝난 후 scheduler.step() 호출하여 학습률 업데이트

        if epoch % config.train.val_period == 0:
            validate(epoch, model, loss_function, val_loader, config,
                     tensorboard_writer, logger)
        #설정된 val_period에 따라 에포크 진행되며 주기적으로 모델 검증

        if (epoch % config.train.checkpoint_period == 0
                or epoch == config.scheduler.epochs):
            checkpoint_config = {'epoch': epoch, 'config': config.as_dict()}
            checkpointer.save(f'checkpoint_{epoch:04d}', **checkpoint_config)
        #checkpoint_period에 맞춰 주기적으로 모델의 상태 저장, 마지막 에포크도 저장

    tensorboard_writer.close()
    #학습 끝나면 tensorboard_writer 닫아서 기록 완료


if __name__ == '__main__':
    main()
