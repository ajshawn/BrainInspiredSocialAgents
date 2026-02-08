function out = align_trials(event_times, input,window)
cc = bwconncomp(event_times);
onsets = cellfun(@(i) i(1),cc.PixelIdxList);
out = [];
for i = 1:length(onsets)
    if onsets(i)+window(1)<=0 || onsets(i)+window(2)>length(input)
        continue
    end
    out = [out,input(onsets(i)+window(1):onsets(i)+window(2))];
end
end