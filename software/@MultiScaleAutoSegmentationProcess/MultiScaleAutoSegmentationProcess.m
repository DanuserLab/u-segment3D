classdef MultiScaleAutoSegmentationProcess < SegmentationProcess
    % A concrete process multi-scale automatic segmentation
    % Segment a single cell image by combining segmentations.
    % see multiScaleAutoSeg_multiObject.m
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
    
    % Andrew R. Jamieson - Sep 2017

    % Replace wrapper fcn from multiScaleAutoSeg to multiScaleAutoSeg_multiObject
    % as per Jungsik's request to reflect new algorithm in MSA_Seg_multiObject_imDir
    % Qiongjing (Jenny) Zou, Aug 2021
    
    methods
        function obj = MultiScaleAutoSegmentationProcess(owner,varargin)
            
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
                super_args{2} = MultiScaleAutoSegmentationProcess.getName;
                super_args{3} = @multiScaleAutoSeg_multiObject;
                if isempty(funParams)
                    funParams=MultiScaleAutoSegmentationProcess.getDefaultParams(owner,outputDir);
                end
                super_args{4} = funParams;
            end
            
            obj = obj@SegmentationProcess(super_args{:});
        end
        
    end
    methods (Static)
        function name = getName()
            name = 'MSA Segmentation';
        end
        function h = GUI()
            h= @msaSegmentationProcessGUI;
        end
        
        function funParams = getDefaultParams(owner,varargin)
            % Input check
            ip=inputParser;
            ip.addRequired('owner', @(x) isa(x,'MovieData'));
            ip.addOptional('outputDir', owner.outputDirectory_, @ischar);
            ip.parse(owner, varargin{:})
            outputDir = ip.Results.outputDir;
            
            % Set default parameters
            funParams.ChannelIndex = 1:numel(owner.channels_);
            funParams.OutputDirectory = [outputDir  filesep 'MultiScaleAutoSeg_Masks'];
            funParams.ProcessIndex = []; %Default is to use raw images % this will be auto-set to ShadeCorrectionProcess or CropShadeCorrectROIProcess in BiosensorsPackage, see sanityCheck in BiosensorsPackage.
            funParams.tightness = .5; % thresholding
            funParams.ObjectNumber = 1;
            funParams.finalRefinementRadius = 3;

            funParams.useSummationChannel = 0; % if true, then do msa seg on the output of summation channel.

            funParams.numVotes = -1; %tightness and numVotes are exclusive options: if one is chosen, the other is inactive (-1); 
                                    % If numVotes chosen, default is 22, value range is 0<= x <= 42. x should be an integer.

            %% extra parameters not on GUI:
            funParams.imagesOut = 1;
            funParams.figVisible = 'on';
            funParams.MinimumSize = 10; % unit is pixel



        %% Parameters for old wrapper fcn:
            % funParams.type = 'middle';
            % funParams.diagnostics = false;
            %% extra parameters?
            % sigmas = [0 0.66 1 1.66 2.5 4];  % unit: pixel (common scales for xxx by xxx size confocal images)
            % p.MinimumSize = 100;        
            % p.ObjectNumber = 1;
            % p.FillHoles = 1;
        end
    end
end