function [data_norm mn stddev]= normalize_feats(data)

    mn = zeros(1,size(data,2));
    stddev = zeros(1,size(data,2));
    for i=1:size(data,2)
       mn(i) = mean(data(:,i));
       stddev(i) = std(data(:,i));
       if (stddev(i) == 0) stddev(i) = 1; end
       data_norm(:,i) = (data(:,i)-mn(i))/stddev(i);
    end