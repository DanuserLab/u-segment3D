classdef MaskProcess < Process
    % Abstract interface for all mask-related processes (segmentation,
    % morphological operations)
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 
    
    % Sebastien Besson Oct 2011
    
    properties (SetAccess = protected)
        maxIndex_;%For index-image segmentation, specifies the highest index present in any frame of the movie.
    end
    
    methods (Access = protected)
        function obj = MaskProcess(owner,name,funName, funParams,outFilePaths)
            % Constructor of class MaskProcess
            if nargin == 0
                super_args = {};
            else
                super_args{1} = owner;
                super_args{2} = name;
            end
            % Call the superclass constructor - these values are private
            obj = obj@Process(super_args{:});
            
            if nargin > 2
                obj.funName_ = funName;
            end
            if nargin > 3
                obj.funParams_ = funParams;
            end
            if nargin > 4
                if ~isempty(outFilePaths) && numel(outFilePaths) ...
                        ~= numel(owner.channels_) || ~iscell(outFilePaths)
                    error('lccb:set:fatal','User-defined: Mask paths must be a cell-array of the same size as the number of image channels!\n\n');
                end
                obj.outFilePaths_ = outFilePaths;
                
            else
                obj.outFilePaths_ = cell(1,numel(owner.channels_));
            end
        end
    end
    methods

        %Checks if a particular channel has masks
        function maskStatus = checkChannelOutput(obj,iChan)
            
            nChanTot = numel(obj.owner_.channels_);
            if nargin < 2 || isempty(iChan)
                iChan = 1:nChanTot; %Default is to check all channels
            end
            nChan = numel(iChan);
            maskStatus = false(1,nChan);
            if all(obj.checkChanNum(iChan))
                for j = 1:nChan
                    %Check the directory and number of masks in each
                    %channel.
                    if exist(obj.outFilePaths_{iChan(j)},'dir') && ...
                            length(imDir(obj.outFilePaths_{iChan(j)})) == obj.owner_.nFrames_;
                        maskStatus(j) = true;
                    end
                end
            end
            
        end
        function setOutMaskPath(obj,chanNum,maskPath)
            if obj.checkChanNum(chanNum)
                obj.outFilePaths_{1,chanNum} = maskPath;
            else
                error('lccb:set:fatal','Invalid mask channel number for mask path!\n\n');
            end
        end
        function fileNames = getOutMaskFileNames(obj,iChan)
            if obj.checkChanNum(iChan)
                fileNames = cellfun(@(x)(imDir(x)),obj.outFilePaths_(iChan),'UniformOutput',false);
                fileNames = cellfun(@(x)(arrayfun(@(x)(x.name),x,'UniformOutput',false)),fileNames,'UniformOutput',false);
                nIm = cellfun(@(x)(length(x)),fileNames);
                if ~all(nIm == obj.owner_.nFrames_)
                    error('Incorrect number of masks found in one or more channels!')
                end
            else
                error('Invalid channel numbers! Must be positive integers less than the number of image channels!')
            end
            
            
        end
        
        function mask = loadChannelOutput(obj, iChan, iFrame, varargin)
            % Input check
            ip =inputParser;
            ip.addRequired('obj');
            ip.addRequired('iChan', @obj.checkChanNum);
            ip.addRequired('iFrame', @obj.checkFrameNum);
%             if obj.owner_.is3D()
                ip.addOptional('iZ', [], @obj.checkDepthNum);
%             end
            ip.addParamValue('output',[],@ischar);            
            ip.parse(obj,iChan,iFrame,varargin{:})
            maskNames = obj.getOutMaskFileNames(iChan);
            if obj.getOwner().is3D()
                %iZ = ip.Results.iZ;
                if ~exist('iZ', 'var')
                    mask = tif3Dread([obj.outFilePaths_{iChan} filesep maskNames{1}{iFrame}]);
                else 
                    mask =imread([obj.outFilePaths_{iChan} filesep maskNames{1}{iFrame}], iZ);
                end
            else
                mask =imread([obj.outFilePaths_{iChan} filesep maskNames{1}{iFrame}]);
            end
%             mask=cell(size(iChan));
%             for i=iChan
%                 maskNames = obj.getOutMaskFileNames(i);
%                 mask{i} = arrayfun(@(j) imread([obj.outFilePaths_{i} filesep...
%                     maskNames{1}{j}]),iFrame,'Unif',0);
%             end
        end
        function output = getDrawableOutput(obj)
            output(1).name='Masks';
            output(1).var='mask';            
            output(1).type='overlay';
            if isempty(obj.maxIndex_)            
                output(1).formatData=@MaskProcess.getMaskBoundaries;
                colors = hsv(numel(obj.owner_.channels_));
                output(1).defaultDisplayMethod=@(x) LineDisplay('Color',colors(x,:));                                
            else
                cMap = randomColormap(obj.maxIndex_,42);%random colors since index is arbitrary, but constant seed so we are consistent across frames.
                output(1).formatData=@(x)(MaskProcess.getMaskOverlayImage(x,cMap));
                %If index values for diff channels are to be differentiated
                %they must be done at the level of the indexes.
                output(1).defaultDisplayMethod=@ImageOverlayDisplay;
                
                output(2).name='Object Number';
                output(2).var = 'number';
                output(2).type = 'overlay';
                output(2).formatData=@(x)(MaskProcess.getObjectNumberText(x,cMap));
                output(2).defaultDisplayMethod=@TextDisplay;
                
                
            end
            
            n = length(output)+1;
            
            output(n).name = 'Mask';
            output(n).formatData = @(x) double(x);
            output(n).var = 'mask_img';
            output(n).type = 'image';
            output(n).defaultDisplayMethod = @ImageDisplay;
        end
        
        function setMaxIndex(obj,maxIndex)
            assert(numel(maxIndex) == 1 && isposint(maxIndex),'Invalid maxIndex! Must be positive integer scalar!')
            obj.maxIndex_ = maxIndex;
        end
        
        function maxIndex = getMaxIndex(obj)            
            maxIndex = obj.maxIndex_;
        end
        
    end
    methods(Static)
        function procClasses = getConcreteClasses()
            % List concrete mask classes
            procClasses = ...
                {'ThresholdProcess';
                 'MSSSegmentationProcess';
                 'ThresholdProcess3D'};
        end
        
        function overlayIm = getMaskOverlayImage(mask,cMap)
            
            overlayIm = zeros([size(mask) 4]);
            overlayIm(:,:,1:3) = ind2rgb(double(mask),cMap);%Convert to double to avoid ind2rgb adding 1            
            overlayIm(:,:,4) = double(mask>0) * .5;%So it works with binary and indexed images                        
            
        end        
        
        function boundaries = getMaskBoundaries(mask)
            % Format mask boundaries in xy-coordinate system
            b=bwboundaries(mask);
            b2 =cellfun(@(x) vertcat(x,[NaN NaN]),b,'Unif',false);
            boundaries =vertcat(b2{:});
            if ~isempty(boundaries)
                boundaries = boundaries(:,2:-1:1);
            else
                boundaries = [NaN NaN];
            end
        end
        
        function out = getObjectNumberText(mask,cMap)
                       
            out.ind = unique(mask(mask(:)>0));
            out.Color = cMap(out.ind,:);
            out.String = arrayfun(@num2str,out.ind,'Unif',0);
            rp = regionprops(mask,'Centroid');            
            out.Position = vertcat(rp(out.ind).Centroid);%regionprops pads zeros if indices are missing
        
        end
    end
end