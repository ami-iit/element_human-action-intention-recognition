function [jointPos, grf, subjectMass, riskMeasure] = extractData(data)

jointPos_tmp = data.kinematicData.jointPos;

grf_tmp= [data.wrenchData.LeftFoot,data.wrenchData.RightFoot]; %ground reaction forces

grf_tmp2= resample(grf_tmp, size(jointPos_tmp,1),  size(grf_tmp,1) );


jointPos= jointPos_tmp (data.startSampleIndex:data.endSampleIndex,:);
grf=grf_tmp2(data.startSampleIndex:data.endSampleIndex,:);
subjectMass= data.mass;
riskMeasure= data.LI;

end

