classdef ThresholdProcess < SegmentationProcess & NonSingularProcess
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
        function obj = ThresholdProcess(owner,varargin)
            
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
                super_args{2} = ThresholdProcess.getName;
                super_args{3} = @thresholdMovie;
                if isempty(funParams)
                    funParams = ThresholdProcess.getDefaultParams(owner,outputDir);
                end
                super_args{4} = funParams;
            end
            
            obj = obj@SegmentationProcess(super_args{:});
        end
        function output = getDrawableOutput(obj)
            output = obj.getDrawableOutput@SegmentationProcess();
            
            n = length(output)+1;
            output(n).name = 'Threshold Values';
            output(n).var = 'thresholdValues';
            output(n).formatData = @(x) [(1:length(x)).' x(:)];
            output(n).type = 'graph';
            output(n).defaultDisplayMethod = @LineDisplay;
        end
        function mask = loadChannelOutput(obj, iChan, iFrame, varargin)
            ip = inputParser;
            ip.addOptional('iFrame',1,@isnumeric);
            ip.addParameter('output','',@ischar);
            ip.StructExpand = true;
            try
                ip.parse(iFrame,varargin{:});
            
                output = ip.Results.output;
            catch err
                output = '';
            end
            
            switch(output)
                case 'thresholdValues'
                    pfName = 'threshold_values_for_channel_';
                    p = obj.getParameters();
                    out = load([p.OutputDirectory filesep pfName num2str(iChan) '.mat'],'thresholdValues');
                    mask = out.thresholdValues;
                otherwise
                    mask = obj.loadChannelOutput@SegmentationProcess(iChan,iFrame,varargin{:});
            end
        end
    end
    methods (Static)
        function name = getName()
            name = 'Thresholding';
        end
        function h = GUI()
            h = @thresholdProcessGUI;
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
            
            % Detect other threshold processes
            isThresholdProcess = cellfun(@(p) isa(p,'ThresholdProcess'),owner.processes_);
            
            % Set default parameters
            funParams.ChannelIndex = 1:numel(owner.channels_);
            if(all(~isThresholdProcess))
                funParams.OutputDirectory = [outputDir  filesep 'masks'];
            else
                % Number mask directories by expected process index
                funParams.OutputDirectory = [outputDir  filesep 'masks' num2str(length(owner.processes_)+1)];
            end
            funParams.ProcessIndex = [];%Default is to use raw images % this will be auto-set to ShadeCorrectionProcess or CropShadeCorrectROIProcess in BiosensorsPackage, see sanityCheck in BiosensorsPackage.
            funParams.PreThreshold = false; % use fixed threshold before automatic threshold
            funParams.ThresholdValue = []; % automatic threshold selection
            funParams.IsPercentile = []; % use percentile rather than absolute
            funParams.MaxJump = 0; %Default is no jump suppression
            funParams.GaussFilterSigma = 0; %Default is no filtering.
            funParams.BatchMode = false;
            funParams.MethodIndx = 1;
        end
    end
end