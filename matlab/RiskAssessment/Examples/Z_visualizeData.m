% Borken/incomplete script file

joints = kinematicData.jointPos_raw;

size_ = size(joints);
time= 1:size_(1);
for i=1:48
    figure; 
    hold on;
    title(kinematicData.jointsName(i))
    plot(time, joints(:,i), 'lineWidth',2);
    plot(time(info.startSampleIndex), joints(info.startSampleIndex, i), '*', 'MarkerSize',10)
    plot(time(info.endSampleIndex), joints(info.endSampleIndex, i), '*','MarkerSize',10)
end

size_ = size(measurements.wrenchData.LeftFoot);
time= 1:size_(1);
for i=1:6
    figure; 
    hold on;
    title(i)
    plot(time, measurements.wrenchData.LeftFoot(:,i), 'lineWidth',2);
    plot(time(info.startSampleIndex), measurements.wrenchData.LeftFoot(info.startSampleIndex, i), '*', 'MarkerSize',10)
    plot(time(info.endSampleIndex), measurements.wrenchData.LeftFoot(info.endSampleIndex, i), '*','MarkerSize',10)
end
