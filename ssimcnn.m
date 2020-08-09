function [score] = ssimcnn(imgRef, imgDist, net, layers)

    numberOfLayers = numel(layers);
    layerwise = zeros(1,numberOfLayers);
    
    for i=1:numberOfLayers
        actRef = activations(net, imgRef, layers{i});
        actDist= activations(net, imgDist,layers{i});
        
        depth=size(actRef, 3);
        subscores=zeros(1,depth);
        
        for j=1:depth
            subscores(j) = ssim(actDist(:,:,j), actRef(:,:,j));
        end
        
        layerwise(i) = mean(subscores(:));
        
    end

    score = abs(prod(layerwise).^(1/numberOfLayers));
    
end

