clear;

%% parameters

% half window size - determines the minimum distance between used features
% points as well as the size of the window around a feature point which is
% used for the color adjustment calculations
hws = 30; 

% image blur size used to reduce the effect of noise
blur_size = 10;

% degrees of polynomials which are fitted to the measured coefficients for
% the Y, Cb and Cr channels
poly_degree_y = 7;
poly_degree_cb_cr = 1;


%% prepare data

% load images
img_1 = imread('Translated.jpg');
img_2 = imread('Warped_Overtune.jpg');

% convert to grayscale
img_1_gray = rgb2gray(img_1);
img_2_gray = rgb2gray(img_2); 


%% feature extraction and matching

% get SURF features
features_1 = detectSURFFeatures(img_1_gray);
features_2 = detectSURFFeatures(img_2_gray);

% extract features/points
[f_1,points_1] = extractFeatures(img_1_gray,features_1);
[f_2,points_2] = extractFeatures(img_2_gray,features_2);

% match features
indexPairs = matchFeatures(f_1,f_2);
matched_1 = points_1(indexPairs(:,1),:);
matched_2 = points_2(indexPairs(:,2),:);

% filter matches according to point distances
valid = [];
for i = 1:size(indexPairs,1)
	% non-max suppression: check if there already is a valid feature
	% within a hws px radius
	is_max = true;
	for j = 1:size(valid,2)
	    diff = matched_1(i).Location - matched_1(valid(j)).Location;
	    dist = diff * diff';
	    if dist < hws^2
            is_max = false;
            break;
	    end
	end
	if is_max && matched_1(i).Scale < 5
	    % save new feature index
	    valid = [valid, i];
	end
end
matched_1 = matched_1(valid);
matched_2 = matched_2(valid);


%% calculate color correction transform

color_factors = double(zeros(256,3));
factor_counter = double(zeros(256,3));

% convert to ycbcr space
img_1_ycbcr = rgb2ycbcr(img_1);
img_2_ycbcr = rgb2ycbcr(img_2);

% smooth images to reduce noise effect
img_1_ycbcr_smooth = imgaussfilt(img_1_ycbcr,blur_size);
img_2_ycbcr_smooth = imgaussfilt(img_2_ycbcr,blur_size);

for i = 1:size(valid,2)
    % collect samples around feature points
    p_1 = round(matched_1(i).Location);
    p_2 = round(matched_2(i).Location);

    hws_1 = min(min(p_1(1)-1, size(img_1_ycbcr_smooth,1) - p_1(1)), min(p_1(2)-1, size(img_1_ycbcr_smooth, 2) - p_1(2)));
    hws_2 = min(min(p_2(1)-1, size(img_2_ycbcr_smooth,1) - p_2(1)), min(p_2(2)-1, size(img_2_ycbcr_smooth, 2) - p_2(2)));
    curr_hws = min(hws, min(hws_1, hws_2));

    img_1_section = img_1_ycbcr_smooth(p_1(2)-curr_hws:p_1(2)+curr_hws, p_1(1)-curr_hws:p_1(1)+curr_hws, :);
    img_2_section = img_2_ycbcr_smooth(p_2(2)-curr_hws:p_2(2)+curr_hws, p_2(1)-curr_hws:p_2(1)+curr_hws, :);

    % calculate the color difference
    color_diff = (double(img_1_section) ./ double(img_2_section));
    
    % save the factors for the respective y, cb and cr channels
    for row = 1:2*curr_hws+1
        for col = 1:2*curr_hws+1
            idx = min(235,max(16,round(img_2_section(row,col,1))));
                color_factors(idx, 1) = ...
                    color_factors(idx, 1) + color_diff(row,col,1);
                factor_counter(idx, 1) = ...
                    factor_counter(idx, 1) + 1;
            for chan = 2:3
                idx = min(240,max(16,round(img_2_section(row,col,chan))));
                color_factors(idx, chan) = ...
                    color_factors(idx, chan) + color_diff(row,col,chan);
                factor_counter(idx, chan) = ...
                    factor_counter(idx, chan) + 1;
            end
        end
    end
end
color_factors = color_factors ./ factor_counter;

% some bins might be empty -> generate missing factors via polynomial
% interpolation
color_factors(isnan(color_factors)) = 0;

[~,Y]=meshgrid(1:3,1:256);
p_y = polyfit(Y(color_factors(:,1)>0,1),color_factors(color_factors(:,1)>0,1), poly_degree_y);
p_cb = polyfit(Y(color_factors(:,2)>0,1),color_factors(color_factors(:,2)>0,2), poly_degree_cb_cr);
p_cr = polyfit(Y(color_factors(:,3)>0,1),color_factors(color_factors(:,3)>0,3), poly_degree_cb_cr);

figure(1); subplot(3,1,1); plot(linspace(1,256,256),color_factors(:,1)); hold on;
subplot(3,1,2); plot(linspace(1,256,256),color_factors(:,2)); hold on;
subplot(3,1,3); plot(linspace(1,256,256),color_factors(:,3)); hold on;

range_y = 16:1:235;
color_factors(range_y,1) = polyval(p_y, range_y);
range_cb_cr = 16:1:240;
color_factors(range_cb_cr,2) = polyval(p_cb, range_cb_cr);
color_factors(range_cb_cr,3) = polyval(p_cr, range_cb_cr);

color_factors(color_factors < 0) = 0;
color_factors(color_factors > 10) = 10;

figure(1); subplot(3,1,1); plot(linspace(1,256,256),color_factors(:,1));
subplot(3,1,2); plot(linspace(1,256,256),color_factors(:,2));
subplot(3,1,3); plot(linspace(1,256,256),color_factors(:,3));


%% apply color correction

% create a ycbcr color map to apply to img_2
color_maps = min(256,max(1,round(color_factors(:,:) .* Y)));

% apply color map
img_2_new(:,:,1) = reshape(color_maps(img_2_ycbcr(:,:,1),1),size(img_2_ycbcr(:,:,1)));
img_2_new(:,:,2) = reshape(color_maps(img_2_ycbcr(:,:,2),2),size(img_2_ycbcr(:,:,2)));
img_2_new(:,:,3) = reshape(color_maps(img_2_ycbcr(:,:,3),3),size(img_2_ycbcr(:,:,3)));

% convert to rgb
img_2_new = ycbcr2rgb(uint8(img_2_new));


%% output

% plot features onto images
set(groot,'defaultLineLineWidth',2.0)
figure(2);
imshow(img_1); hold on;
plot(matched_1);
img_1 = getframe;

imshow(img_2); hold on;
plot(matched_2);
img_2 = getframe;

imshow(img_2_new); hold on;
plot(matched_2);
img_2_new = getframe;
close;

figure(2);
montage({img_2.cdata,img_1.cdata,img_2_new.cdata,img_1.cdata});
