classdef MovieList < MovieObject
    % Concrete implementation of MovieObject for a list of movies
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
    
    properties
        movieListFileName_   % The name under which the movie list is saved
        movieListPath_       % The path where the movie list is saved
    end
    properties (SetAccess = protected)
        movieDataFile_       % Cell array of movie data's directory
    end
    properties(Transient = true);
        movies_              % Cell array of movies
    end
    
    methods
        function obj = MovieList(movies, outputDirectory, varargin)
            % Constructor for the MovieList object
            
            if nargin > 0
                if iscellstr(movies)
                    obj.movieDataFile_ = movies(:)';
                elseif iscell(movies) && all(cellfun(@(x)isa(x,'MovieData'),movies))
                    obj.movieDataFile_ = cellfun(@getFullPath,...
                        movies,'UniformOutput',false);
                    obj.movies_ = movies;
                elseif isa(movies, 'MovieData')
                    obj.movieDataFile_ = arrayfun(@getFullPath,...
                        movies,'UniformOutput',false);
                    obj.movies_ = num2cell(movies);
                else
                    error('lccb:ml:constructor','Movies should be a cell array or a array of MovieData');
                end
                
                obj.outputDirectory_ = outputDirectory;
                if ~isempty(obj.outputDirectory_)
                    obj.setPath(obj.outputDirectory_)
                end
                
                % Construct the Channel object
                nVarargin = numel(varargin);
                if nVarargin > 1 && mod(nVarargin,2)==0
                    for i=1 : 2 : nVarargin-1
                        obj.(varargin{i}) = varargin{i+1};
                    end
                end
                obj.createTime_ = clock;
            end
        end
        
        
        %%  Set/get methods

        function set.movieListPath_(obj, path)
            % Set movie list path
            endingFilesepToken = [regexptranslate('escape',filesep) '$'];
            path = regexprep(path,endingFilesepToken,'');
            obj.checkPropertyValue('movieListPath_',path);
            obj.movieListPath_ = path;
        end
        
        function set.movieListFileName_(obj, filename)
            obj.checkPropertyValue('movieListFileName_',filename);
            obj.movieListFileName_ = filename;
        end
               
        function movies = getMovies(obj,varargin)
            % Get the movies from a movie list
            
            ip =inputParser;
            allIndex = 1 : obj.getSize();
            ip.addOptional('index',allIndex,@(x) all(ismember(x,allIndex)));
            ip.parse(varargin{:});
            
            if isempty(obj.movies_), obj.sanityCheck; end
            movies = obj.movies_(ip.Results.index);
        end
        
        function movie = getMovie(obj, i)
            % Get the movies from a movie list
            
            assert(isscalar(i) && ismember(i, 1 : obj.getSize()));
            movie = obj.movies_{i};
        end
        
        function size = getSize(obj)
            % Get the movies from a movie list
            
            size = numel(obj.movieDataFile_);
        end
        %% Sanity check/relocation
        function movieException = sanityCheck(obj, varargin)
            % Check the sanity of the MovieData objects
            %
            % First call the superclass sanityCheck. Then load the individual 
            % movies in the list (runs sanityCheck on each movie).
            % Save the movie list to disk if run successfully.
            
            % Call the superclass sanityCheck
            askUser = sanityCheck@MovieObject(obj, varargin{:});
            
            % Load movie components (run sanityCheck on each of them)
            movieException = cell(1, obj.getSize());
            for i = 1 : obj.getSize()
                try
                    if i <= numel(obj.movies_) && ~isempty(obj.getMovie(i))
                        [moviePath,movieName,movieExt] = fileparts(obj.movieDataFile_{i});
                        obj.getMovie(i).sanityCheck(moviePath,[movieName movieExt], askUser);
                    else
                        fprintf(1,'Loading movie %g/%g\n',i,obj.getSize());
                        if obj.isOmero() && obj.canUpload()
                            [~, id] =fileparts(fileparts(obj.movieDataFile_{i}));
                            obj.movies_{i} = MovieData.load(obj.getOmeroSession(),...
                                str2double(id), askUser);
                        else
                            obj.movies_{i} = MovieData.load(obj.movieDataFile_{i}, askUser);
                        end
                        
                        % Check for other movies in the graph
                        otherROIs = obj.movies_{i}.getAncestor().getDescendants();
                        if ~isempty(otherROIs),
                            obj.attachMovies(otherROIs, i + 1 : obj.getSize());
                        end
                    end
                catch ME
                    movieException{i} = ME;
                    continue
                end
            end
            
            % Throw exception if at least one movie failed during loading
            if ~all(cellfun(@isempty,movieException)),
                ME = MException('lccb:ml:sanitycheck','Failed to load movie(s)');
                for i=find(~cellfun(@isempty,movieException));
                    ME = ME.addCause(movieException{i});
                end
                throw(ME);
            end
            
            disp('Saving movie list');
            obj.save();
        end
        
        function attachMovies(obj, movies, varargin)
            % attachMovies attaches one or several movies to a list

            % Input check
            fullrange = 1:obj.getSize();
            ip = inputParser;
            ip.addRequired('movies', @(x) isa(x, 'MovieData'));
            ip.addOptional('range', fullrange, @(x) all(ismember(x, fullrange)));
            ip.parse(movies, varargin{:});
            
            for movie = movies
                canAttach = strcmp(movie.getFullPath(),...
                    obj.movieDataFile_(ip.Results.range));
                if any(canAttach)
                    j = ip.Results.range(find(canAttach, 1));
                    fprintf(1,'Attaching movie %g/%g\n',j,obj.getSize());
                    obj.movies_{j} = movie;
                end
            end
        end
        
        function relocate(obj,oldRootDir,newRootDir,full)
            % Relocate  analysis
            relocate@MovieObject(obj,oldRootDir,newRootDir);            
            if nargin<3 || ~full, return; end
            
            % Relocate the movie paths
            fprintf(1,'Relocating movies from %s to %s\n',oldRootDir,newRootDir);
            for i = 1 : obj.getSize()
                obj.movieDataFile_{i} = relocatePath(obj.movieDataFile_{i},oldRootDir,newRootDir);
            end
        end
        
        function save(ML,varargin)
            
            % Check path validity for movie list
            fullPath = ML.getFullPath();
            assert(~isempty(fullPath), 'Invalid path');
            
            % Backup existing file and save the movie list
            try
            if exist(fullPath,'file')
                movefile(fullPath,[fullPath(1:end-3) 'old' '.' datestr(now,'yyyymmddTHHMMSS')],'f');
            end
            save(fullPath, 'ML');
            catch err
                disp(getReport(err))
            end
            
            % Save to OMERO if OMERO object
            if ML.isOmero() && ML.canUpload(),
                omeroSave(ML);
            end
        end
        
        % Override arithmetic operations to faciliate combining movie lists
        C = plus(A,B,outputDirectory);
        C = minus(A,B,outputDirectory);
        
    end
    
    methods(Static)
        
        function obj = load(varargin)
            % Load or a movie list
            
            if(nargin == 0)
                %If called with no arguments prompt for a .mat file
                [filename,pathname] = uigetfile('*.mat','Select .mat file containing a MovieList object');
                if(filename)
                    varargin{1} = [pathname filesep filename];
                else
                    obj = MovieList.empty;
                    return;
                end
            end
%             assert(MovieList.isOmeroSession(varargin{1}) || ...
%                 exist(varargin{1}, 'file') == 2)
            
            if MovieList.isOmeroSession(varargin{1}),
                obj = MovieList.loadOmero(varargin{:});
            else
                assert(strcmpi(varargin{1}(end-3:end), '.mat'),...
                    'Input must be a MAT file');
                % loadMatFile will catch if file does not exist
                [obj, filepath] = MovieList.loadMatFile(varargin{1});
                
                % Perform sanityCheck using the input path
                [moviePath,movieName,movieExt]=fileparts(filepath);
                obj.sanityCheck(moviePath,[movieName movieExt], varargin{2:end});
            end
        end
        
        function obj = loadOmero(session, varargin)
            % Load a movie list from a dataset stored onto an OMERO server
            obj = getOmeroLists(session, varargin{:});
        end
        
        function [obj, filepath] = loadMatFile(filepath)
            % Load a movie list from a local MAT file
            [obj, filepath] = MovieObject.loadMatFile('MovieList', filepath);
            obj.movies_=cell(1,length(obj.movieDataFile_));
            for mIdx=1:length(obj.movieDataFile_)
                if(exist(obj.movieDataFile_{mIdx}))
                    obj.movies_{mIdx}=MovieData.loadMatFile(obj.movieDataFile_{mIdx});
                end
            end
        end
        
        function status=checkValue(property,value)
           % Return true/false if the value for a given property is valid
            
           % Parse input
           ip = inputParser;
           ip.addRequired('property',@(x) ischar(x) || iscell(x));
           ip.parse(property);
           if iscell(property)
               ip.addRequired('value',@(x) iscell(x)&&isequal(size(x),size(property)));
               ip.parse(property,value);
               status=cellfun(@(x,y) MovieList.checkValue(x,y),property,value);
               return
           end
           
           % Get validator for single property
           validator=MovieList.getPropertyValidator(property);
           propName = regexprep(regexprep(property,'(_\>)',''),'([A-Z])',' ${lower($1)}');
           assert(~isempty(validator),['No validator defined for property ' propName]);
           
           % Return result of validation
           status = isempty(value) || validator(value);
        end
        
        function validator = getPropertyValidator(property) 
            validator = getPropertyValidator@MovieObject(property);
            if ~isempty(validator), return; end
            if ismember(property, {'movieListPath_','movieListFileName_'})
                validator=@ischar;
            end
        end
        
        function propName = getPathProperty()
            propName = 'movieListPath_';
        end
        function propName = getFilenameProperty()
            propName = 'movieListFileName_';
        end
        
        % Use Regular expression to build a movie list by scanning dirs
        ML = buildByRegexp(filter,outputDirectory,findMatsInDirs);
    end
end