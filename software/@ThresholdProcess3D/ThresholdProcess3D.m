classdef ThresholdProcess3D < SegmentationProcess
    %A function-specific process for segmenting via thresholding using
    %thresholdMovie.m
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
    
    methods (Access = public)
        function obj = ThresholdProcess3D(owner,varargin)
            
            if nargin == 0
                super_args = {};
            else
                % Input check
                ip = inputParser;
                ip.addRequired('owner',@(x) isa(x,'MovieData'));
                ip.addOptional('outputDir',owner.outputDirectory_,@ischar);
                ip.addOptional('funParams',[],@isstruct);
                ip.parse(owner,varargin{:});
                outputDir = ip.Results.outputDir;
                funParams = ip.Results.funParams;
                
                % Define arguments for superclass constructor
                super_args{1} = owner;
                super_args{2} = ThresholdProcess3D.getName;
                super_args{3} = @threshold3DMovie;
                if isempty(funParams)
                    funParams = ThresholdProcess.getDefaultParams(owner,outputDir);
                end
                super_args{4} = funParams;
            end
            
            obj = obj@SegmentationProcess(super_args{:});
        end
        
        function drawImaris(obj,iceConn)
            
            for iChan = obj.getParameters.ChannelIndex
                
                if obj.checkChannelOutput(iChan)
                    for iFrame = 1:obj.owner_.nFrames_;
                        mask = obj.loadChannelOutput(iChan,iFrame);
                        if nnz(mask) > 0
                            mSurf = isosurface(mask,.5);
                            norms = isonormals(mask,mSurf.vertices);
                            norms = -norms;%Imaris uses opposite convention for normals
                            mSurf.vertices(:,1:2) = mSurf.vertices(:,[2 1]) * obj.owner_.pixelSize_ / 1e3;%Swap X and Y to agree with imaris coord
                            mSurf.vertices(:,3) = mSurf.vertices(:,3) * obj.owner_.pixelSizeZ_ / 1e3;

                            dataSet = iceConn.mImarisApplication.GetDataSet;
                            chanCol = iceConn.mapRgbaScalarToVector(dataSet.GetChannelColorRGBA(iChan-1));                                        
                            iceConn.createAndSetSurfaces(mSurf,norms,iFrame-1,['Mask ' char(dataSet.GetChannelName(iChan-1))],chanCol);
                        end
                    end
                end
            end
            
        end
        
        
    end
    methods (Static)
        function name = getName()
            name = '3D Thresholding';
        end
        function h = GUI()
            h= @thresholdProcessGUI;
        end
        function methods = getMethods(varargin)
            thresholdingMethods(1).name = 'MinMax';
            thresholdingMethods(1).func = @thresholdFluorescenceImage;
            thresholdingMethods(2).name = 'Otsu';
            thresholdingMethods(2).func = @thresholdOtsu;
            thresholdingMethods(3).name = 'Rosin';
            thresholdingMethods(3).func = @thresholdRosin;
            thresholdingMethods(4).name = 'Gradient-based';
            thresholdingMethods(4).func = @intensityBinnedGradientThreshold;
            
            ip=inputParser;
            ip.addOptional('index',1:length(thresholdingMethods),@isvector);
            ip.parse(varargin{:});
            index = ip.Results.index;
            methods=thresholdingMethods(index);
        end
        
        function funParams = getDefaultParams(owner,varargin)
            % Input check
            ip=inputParser;
            ip.addRequired('owner',@(x) isa(x,'MovieData'));
            ip.addOptional('outputDir',owner.outputDirectory_,@ischar);
            ip.parse(owner, varargin{:})
            outputDir=ip.Results.outputDir;
            
            % Set default parameters
            funParams.ChannelIndex = 1:numel(owner.channels_);
            funParams.OutputDirectory = [outputDir  filesep 'masks'];
            funParams.ProcessIndex = [];%Default is to use raw images
            funParams.ThresholdValue = []; % automatic threshold selection
            funParams.MaxJump = 0; %Default is no jump suppression
            funParams.GaussFilterSigma = 0; %Default is no filtering.
            funParams.BatchMode = false;
            funParams.MethodIndx = 1;
        end

        
%         function mask = loadChannelOutput(obj, iChan,iFrame,varargin)
%             %% temporarily placed here, might need to move to MaskProcess!!
%             % Input check
%             ip =inputParser;
%             ip.addRequired('obj');
%             ip.addRequired('iChan',@(x) ismember(x,1:numel(obj.owner_.channels_)));
%             ip.addRequired('iFrame',@(x) ismember(x,1:obj.owner_.nFrames_));
%             ip.addOptional('iZ',@(x) ismember(x,1:obj.owner_.zSize_));
%             ip.addParamValue('output',[],@ischar);
%             ip.parse(obj,iChan,iFrame,varargin{:})
%             iZ = ip.Results.iZ;
%             
%             
%             % Data loading
%             maskNames = obj.getOutMaskFileNames(iChan);
%             mask =imread([obj.outFilePaths_{iChan} filesep maskNames{1}{iFrame}], iZ);
%             %             mask=cell(size(iChan));
%             %             for i=iChan
%             %                 maskNames = obj.getOutMaskFileNames(i);
%             %                 mask{i} = arrayfun(@(j) imread([obj.outFilePaths_{i} filesep...
%             %                     maskNames{1}{j}]),iFrame,'Unif',0);
%             %             end
%         end
    end
end