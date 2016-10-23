function [data,target,splitarr]=splitData(oldData,oldtarget,splitindex)
fn=unique(oldData(splitindex,:));
data=cell(length(fn),1);
target=cell(length(fn),1);
splitarr=zeros(size(fn));
for i=1:length(fn)
    fcolumn=find(oldData(splitindex,:)==fn(i));
    data(i) =oldData(:,fcolumn);
    target(i) = oldtarget(:,fcolumn);
    data{i}(splitindex,:)=[];
    splitarr(i)=fn(i);
end    
end