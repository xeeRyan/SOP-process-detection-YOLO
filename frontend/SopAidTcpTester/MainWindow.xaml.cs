using System.Diagnostics;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Runtime.CompilerServices;
using Microsoft.Win32;
using Rectangle = System.Windows.Shapes.Rectangle;

namespace SopAidTcpTester;

public partial class MainWindow : Window
{
    private static readonly JsonSerializerOptions PrettyJson = new() { WriteIndented = true };
    private Process? _serverProcess;
    private BitmapImage? _roiBitmap;
    private Point? _roiDragStart;
    private Rectangle? _roiDraftRectangle;
    private bool _trainingActive;
    private string? _activeTrainingTaskId;
    private string? _activeTrainingProjectDir;
    private static readonly Brush[] RoiColors = [Brushes.LimeGreen, Brushes.Orange, Brushes.DeepSkyBlue, Brushes.Magenta, Brushes.Gold];
    public ObservableCollection<WorkflowStepItem> WorkflowSteps { get; } = [];
    public ObservableCollection<string> AvailableClassNames { get; } = [];
    public ObservableCollection<string> AvailableRoiIds { get; } = [];
    public IReadOnlyList<string> TriggerTypes { get; } =
    [
        "object_in_roi", "object_present", "hand_in_roi", "object_event",
        "composite", "object_count", "object_transition", "duration"
    ];
    public IReadOnlyList<string> EventTypes { get; } =
    [
        "", "object_appear", "object_disappear", "object_enter_roi",
        "object_exit_roi", "object_move_roi"
    ];

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;
        PackageRootTextBox.Text = FindPackageRoot();
        ProjectDirTextBox.Text = Path.Combine(PackageRootTextBox.Text, "projects", "SK_DEMO");
        LoadRoiSample();
        LoadWorkflowSample();
        LoadWorkflowEditorFromJson();
        RequestTextBox.Text = FormatJson(new { command = "health" });
        AppendLog("前端联调 Demo 已启动。请先检查算法服务。");
        RefreshModelChoices();
        RefreshConversionModelChoices();
        RefreshDatasetChoices();
    }

    private async void HealthButton_Click(object sender, RoutedEventArgs e)
    {
        var response = await SendCommandAsync(new { command = "health" });
        var ok = response?["status"]?.GetValue<string>() == "ok";
        var supportsTrainingControl = HasCapability(response, "training_stop_after_epoch");
        ServiceIndicator.Fill = new SolidColorBrush(ok ? Color.FromRgb(34, 197, 94) : Color.FromRgb(239, 68, 68));
        ServiceStatusText.Text = ok && supportsTrainingControl ? "服务在线" : ok ? "旧版服务" : "服务异常";
        if (ok && !supportsTrainingControl)
        {
            AppendLog("当前9000端口连接的是旧版算法服务，不支持训练进度和停止控制。请先关闭旧服务再启动新版。");
        }
    }

    private async void ListProjectsButton_Click(object sender, RoutedEventArgs e)
    {
        var response = await SendCommandAsync(new { command = "list_projects" });
        var data = GetData(response);
        if (data?["projects"] is not JsonArray projects)
        {
            return;
        }

        ProjectsListBox.Items.Clear();
        foreach (var node in projects.OfType<JsonObject>())
        {
            var id = node["project_id"]?.ToString() ?? "<unknown>";
            var name = node["name"]?.ToString() ?? id;
            var status = node["status"]?.ToString() ?? "draft";
            var ready = node["ready"]?.GetValue<bool>() ?? false;
            var dir = node["project_dir"]?.ToString() ?? string.Empty;
            ProjectsListBox.Items.Add(new ProjectListItem(id, name, status, ready, dir));
        }
    }

    private async void CreateProjectButton_Click(object sender, RoutedEventArgs e)
    {
        var classes = ParseClasses();
        var response = await SendCommandAsync(new
        {
            command = "create_project",
            params_ = new
            {
                project_id = NewProjectIdTextBox.Text.Trim(),
                name = NewProjectNameTextBox.Text.Trim(),
                description = ProjectDescriptionTextBox.Text.Trim(),
                classes,
            },
        }, renameParams: true);
        UpdateProjectSummary(response);
    }

    private async void GetProjectButton_Click(object sender, RoutedEventArgs e)
    {
        var response = await SendCommandAsync(ProjectCommand("get_project"));
        UpdateProjectSummary(response, loadEditors: true);
    }

    private async void ActivateProjectButton_Click(object sender, RoutedEventArgs e)
    {
        var response = await SendCommandAsync(ProjectCommand("activate_project"));
        UpdateProjectSummary(response);
    }

    private async void SaveRoisButton_Click(object sender, RoutedEventArgs e)
    {
        var rois = ParseObject(RoisJsonTextBox.Text, "ROI JSON");
        if (rois is null) return;
        var response = await SendCommandAsync(new
        {
            command = "save_rois",
            @params = new { sop_project_dir = GetProjectDir(), rois },
        });
        UpdateProjectSummary(response);
    }

    private async void SaveWorkflowButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            WorkflowJsonTextBox.Text = BuildWorkflowJsonFromEditor().ToJsonString(PrettyJson);
        }
        catch (Exception ex)
        {
            AppendLog("流程步骤配置错误: " + ex.Message);
            return;
        }
        var workflow = ParseObject(WorkflowJsonTextBox.Text, "流程 JSON");
        if (workflow is null) return;
        var response = await SendCommandAsync(new
        {
            command = "save_workflow",
            @params = new { sop_project_dir = GetProjectDir(), workflow },
        });
        UpdateProjectSummary(response);
    }

    private async void ExtractFramesButton_Click(object sender, RoutedEventArgs e)
    {
        var videoPaths = Lines(VideoPathsTextBox.Text).ToArray();
        await SendCommandAsync(new
        {
            command = "extract_frames",
            @params = new
            {
                sop_project_dir = GetProjectDir(),
                video_paths = videoPaths,
                frames_per_second = ParseDouble(FramesPerSecondTextBox, "每秒抽帧数"),
                max_frames_per_video = ParseInt(MaxFramesTextBox, "最大抽帧数"),
                copy_videos = CopyVideosCheckBox.IsChecked == true,
                jpeg_quality = 95,
            },
        });
    }

    private async void ImportLabelsButton_Click(object sender, RoutedEventArgs e)
    {
        await SendCommandAsync(new
        {
            command = "import_annotations",
            @params = new
            {
                sop_project_dir = GetProjectDir(),
                labels_dir = LabelsDirTextBox.Text.Trim(),
                overwrite = OverwriteLabelsCheckBox.IsChecked == true,
                mark_missing_as_empty = MissingAsEmptyCheckBox.IsChecked == true,
            },
        });
    }

    private async void BuildDatasetButton_Click(object sender, RoutedEventArgs e)
    {
        await SendCommandAsync(new
        {
            command = "build_dataset",
            @params = new
            {
                sop_project_dir = GetProjectDir(),
                dataset_name = DatasetNameComboBox.Text.Trim(),
                train_ratio = ParseDouble(TrainRatioTextBox, "训练比例"),
                val_ratio = ParseDouble(ValRatioTextBox, "验证比例"),
                test_ratio = ParseDouble(TestRatioTextBox, "测试比例"),
                seed = 42,
                require_all_annotated = true,
                require_all_splits = true,
            },
        });
        RefreshDatasetChoices();
    }

    private async void TrainProjectButton_Click(object sender, RoutedEventArgs e)
    {
        var health = await SendCommandAsync(new { command = "health" });
        if (!HasCapability(health, "training_stop_after_epoch"))
        {
            AppendLog("已阻止训练：当前算法服务版本过旧，无法提供训练进度和安全停止。请重启新版服务。");
            return;
        }
        var baseModel = BaseModelTextBox.Text.Trim();
        var parameters = new JsonObject
        {
            ["sop_project_dir"] = GetProjectDir(),
            ["dataset_name"] = DatasetNameComboBox.Text.Trim(),
            ["model_id"] = ModelIdTextBox.Text.Trim(),
            ["model_version"] = ModelVersionTextBox.Text.Trim(),
            ["epochs"] = ParseInt(EpochsTextBox, "Epochs"),
            ["batch"] = ParseInt(BatchTextBox, "Batch"),
            ["imgsz"] = ParseInt(ImageSizeTextBox, "Image Size"),
            ["workers"] = ParseInt(WorkersTextBox, "Workers"),
            ["optimizer"] = OptimizerComboBox.Text.Trim(),
            ["amp"] = AmpCheckBox.IsChecked == true,
            ["set_active"] = SetActiveModelCheckBox.IsChecked == true,
        };
        if (!string.IsNullOrWhiteSpace(baseModel)) parameters["base_model_path"] = baseModel;
        SetOptionalText(parameters, "run_name", RunNameTextBox);
        SetOptionalText(parameters, "device", DeviceTextBox);
        SetOptionalDouble(parameters, "hsv_h", HsvHTextBox);
        SetOptionalDouble(parameters, "hsv_s", HsvSTextBox);
        SetOptionalDouble(parameters, "hsv_v", HsvVTextBox);
        SetOptionalDouble(parameters, "degrees", DegreesTextBox);
        SetOptionalDouble(parameters, "translate", TranslateTextBox);
        SetOptionalDouble(parameters, "scale", ScaleTextBox);
        SetOptionalDouble(parameters, "shear", ShearTextBox);
        SetOptionalDouble(parameters, "fliplr", FlipLrTextBox);
        SetOptionalDouble(parameters, "flipud", FlipUdTextBox);
        SetOptionalDouble(parameters, "mosaic", MosaicTextBox);
        SetOptionalDouble(parameters, "mixup", MixupTextBox);
        SetOptionalDouble(parameters, "copy_paste", CopyPasteTextBox);
        _trainingActive = true;
        _activeTrainingTaskId = $"{ModelIdTextBox.Text.Trim()}_{ModelVersionTextBox.Text.Trim()}";
        _activeTrainingProjectDir = GetProjectDir();
        TrainProjectButton.IsEnabled = false;
        StopTrainingButton.IsEnabled = false;
        StopTrainingButton.Content = "完成当前轮后停止";
        TrainingProgressBar.Value = 0;
        TrainingProgressText.Text = "正在启动训练……";
        var requestTask = SendCommandAsync(new JsonObject { ["command"] = "train_project", ["params"] = parameters });
        try
        {
            while (!requestTask.IsCompleted)
            {
                RefreshTrainingStatus();
                await Task.WhenAny(requestTask, Task.Delay(1000));
            }
            await requestTask;
            RefreshTrainingStatus();
            RefreshConversionModelChoices();
        }
        finally
        {
            _trainingActive = false;
            _activeTrainingTaskId = null;
            _activeTrainingProjectDir = null;
            TrainProjectButton.IsEnabled = true;
            StopTrainingButton.IsEnabled = false;
        }
    }

    private void StopTrainingButton_Click(object sender, RoutedEventArgs e)
    {
        if (!_trainingActive) return;
        var taskId = _activeTrainingTaskId;
        var projectDir = _activeTrainingProjectDir;
        if (string.IsNullOrWhiteSpace(taskId) || string.IsNullOrWhiteSpace(projectDir)) return;
        var controlPath = Path.Combine(projectDir, "outputs", "training", "stoptrain.json");
        try
        {
            WriteJsonAtomic(controlPath, new JsonObject
            {
                ["task_id"] = taskId,
                ["is_stop"] = true,
            });
            StopTrainingButton.IsEnabled = false;
            StopTrainingButton.Content = "正在停止……";
            TrainingProgressText.Text = "已请求停止，将在当前轮完成后结束";
        }
        catch (Exception ex)
        {
            AppendLog("停止训练请求写入失败: " + ex.Message);
        }
    }

    private void RefreshTrainingStatus()
    {
        if (string.IsNullOrWhiteSpace(_activeTrainingTaskId) || string.IsNullOrWhiteSpace(_activeTrainingProjectDir)) return;
        var statusPath = Path.Combine(_activeTrainingProjectDir, "outputs", "training", "training_status.json");
        try
        {
            if (!File.Exists(statusPath) || JsonNode.Parse(File.ReadAllText(statusPath)) is not JsonObject status) return;
            var expectedTaskId = _activeTrainingTaskId;
            if (!string.Equals(status["task_id"]?.ToString(), expectedTaskId, StringComparison.Ordinal)) return;
            var current = status["current_epoch"]?.GetValue<int>() ?? 0;
            var total = status["total_epochs"]?.GetValue<int>() ?? 0;
            var progress = status["progress_percent"]?.GetValue<double>() ?? 0;
            var state = status["status"]?.ToString() ?? "running";
            var message = status["message"]?.ToString() ?? string.Empty;
            TrainingProgressBar.Value = Math.Clamp(progress, 0, 100);
            TrainingProgressText.Text = $"{current} / {total}  {message}";
            if (state == "stopping")
            {
                StopTrainingButton.IsEnabled = false;
                StopTrainingButton.Content = "正在停止……";
            }
            else if (state == "running" && _trainingActive)
            {
                StopTrainingButton.IsEnabled = true;
                StopTrainingButton.Content = "完成当前轮后停止";
            }
            else if (state is "completed" or "stopped" or "failed")
            {
                StopTrainingButton.IsEnabled = false;
                StopTrainingButton.Content = "完成当前轮后停止";
            }
        }
        catch (Exception ex) when (ex is IOException or JsonException)
        {
            // The backend replaces the file atomically; retry on the next polling tick.
        }
    }

    private static void WriteJsonAtomic(string path, JsonObject data)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temporary = path + ".tmp";
        File.WriteAllText(temporary, data.ToJsonString(PrettyJson), Encoding.UTF8);
        File.Move(temporary, path, true);
    }

    private async void ConvertProjectModelButton_Click(object sender, RoutedEventArgs e)
    {
        var formats = new JsonArray();
        if (ExportTorchScriptCheckBox.IsChecked == true) formats.Add("torchscript");
        if (ExportOnnxCheckBox.IsChecked == true) formats.Add("onnx");
        if (ExportEngineCheckBox.IsChecked == true) formats.Add("engine");
        if (formats.Count == 0)
        {
            throw new InvalidOperationException("请至少选择一种模型转换格式。");
        }

        if (ConversionModelComboBox.SelectedItem is not ModelVersionChoice selectedModel)
        {
            throw new InvalidOperationException("请先从下拉框选择一个已训练模型。");
        }

        var parameters = new JsonObject
        {
            ["sop_project_dir"] = GetProjectDir(),
            ["model_id"] = selectedModel.ModelId,
            ["model_version"] = selectedModel.ModelVersion,
            ["formats"] = formats,
            ["overwrite"] = ConversionOverwriteCheckBox.IsChecked == true,
        };
        await SendCommandAsync(new JsonObject
        {
            ["command"] = "convert_project_model",
            ["params"] = parameters,
        });
    }

    private async void DetectButton_Click(object sender, RoutedEventArgs e)
    {
        var parameters = new JsonObject
        {
            ["sop_project_dir"] = GetProjectDir(),
            ["video_path"] = DetectVideoTextBox.Text.Trim(),
            ["output_dir"] = DetectOutputTextBox.Text.Trim(),
            ["confidence_threshold"] = ParseDouble(ConfidenceTextBox, "置信度"),
            ["nms_threshold"] = ParseDouble(NmsThresholdTextBox, "NMS 阈值"),
            ["inference_device"] = (InferenceDeviceComboBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "auto",
            ["inference_backend"] = (InferenceBackendComboBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "python",
            ["enable_yolo"] = EnableYoloCheckBox.IsChecked == true,
            ["enable_hand_pose"] = EnableHandPoseCheckBox.IsChecked == true,
            ["hand_pose_sample_interval"] = ParseInt(HandPoseIntervalTextBox, "手部采样间隔"),
            ["output_video"] = OutputVideoCheckBox.IsChecked == true,
            ["tracking_iou_threshold"] = ParseDouble(TrackingIouTextBox, "跟踪 IoU 阈值"),
            ["tracking_max_missing_frames"] = ParseInt(TrackingMissingTextBox, "跟踪最大丢失帧"),
            ["event_lost_tolerance_frames"] = ParseInt(EventLostToleranceTextBox, "事件丢失容忍帧"),
            ["output_json"] = OutputJsonCheckBox.IsChecked == true,
            ["realtime_display"] = RealtimeDisplayCheckBox.IsChecked == true,
        };
        var model = DetectModelComboBox.SelectedItem?.ToString()?.Trim() ?? string.Empty;
        if (!string.IsNullOrWhiteSpace(model)) parameters["model_path"] = model;
        SetOptionalText(parameters, "hand_pose_model_path", HandPoseModelTextBox);
        var targetClasses = TargetClassesTextBox.Text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (targetClasses.Length > 0)
        {
            parameters["target_classes"] = new JsonArray(
                targetClasses.Select(value => (JsonNode?)JsonValue.Create(value)).ToArray()
            );
        }
        SetOptionalJsonObject(parameters, "sop_step_enabled", StepEnabledJsonTextBox);
        SetOptionalJsonObject(parameters, "sop_trigger_sources", TriggerSourcesJsonTextBox);
        SetOptionalJsonObject(parameters, "sop_step_timeouts_sec", StepTimeoutsJsonTextBox);
        await SendCommandAsync(new JsonObject { ["command"] = "detect", ["params"] = parameters });
    }

    private async void SendJsonButton_Click(object sender, RoutedEventArgs e)
    {
        var request = ParseObject(RequestTextBox.Text, "请求 JSON");
        if (request is not null) await SendCommandAsync(request);
    }

    private void LoadHealthButton_Click(object sender, RoutedEventArgs e) => RequestTextBox.Text = FormatJson(new { command = "health" });
    private void LoadGetProjectButton_Click(object sender, RoutedEventArgs e) => RequestTextBox.Text = FormatJson(ProjectCommand("get_project"));
    private void LoadDetectButton_Click(object sender, RoutedEventArgs e) => RequestTextBox.Text = FormatJson(new { command = "detect", @params = new { sop_project_dir = GetProjectDir(), video_path = "videos/sk.mp4", output_dir = "outputs/frontend_detect", realtime_display = false } });
    private void LoadRoiSampleButton_Click(object sender, RoutedEventArgs e) { LoadRoiSample(); RefreshRoiEditor(); }
    private void LoadWorkflowSampleButton_Click(object sender, RoutedEventArgs e) { LoadWorkflowSample(); LoadWorkflowEditorFromJson(); }
    private void LoadWorkflowTableButton_Click(object sender, RoutedEventArgs e) => LoadWorkflowEditorFromJson();

    private void LoadRoiSample()
    {
        RoisJsonTextBox.Text = FormatJson(new
        {
            schema_version = "1.0",
            coordinate_type = "normalized",
            regions = new[]
            {
                new { id = "work", name = "主工作区", shape = "rectangle", points = new[] { new[] { 0.10, 0.10 }, new[] { 0.90, 0.90 } } },
                new { id = "tool_home", name = "工具归位区", shape = "rectangle", points = new[] { new[] { 0.05, 0.10 }, new[] { 0.25, 0.35 } } },
            },
        });
    }

    private void BrowseRoiImageButton_Click(object sender, RoutedEventArgs e)
    {
        var framesDir = Path.Combine(GetProjectDir(), "frames");
        var dialog = new OpenFileDialog
        {
            Title = "选择用于划分 ROI 的抽帧图片",
            Filter = "图片文件|*.jpg;*.jpeg;*.png;*.bmp|所有文件|*.*",
            InitialDirectory = Directory.Exists(framesDir) ? framesDir : GetProjectDir(),
        };
        if (dialog.ShowDialog() != true) return;

        try
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.UriSource = new Uri(dialog.FileName, UriKind.Absolute);
            bitmap.EndInit();
            bitmap.Freeze();
            _roiBitmap = bitmap;
            RoiSourceImage.Source = bitmap;
            RoiImagePlaceholder.Visibility = Visibility.Collapsed;
            RoiImageInfoText.Text = $"{Path.GetFileName(dialog.FileName)} · {bitmap.PixelWidth} × {bitmap.PixelHeight}px";
            RefreshRoiEditor();
        }
        catch (Exception ex)
        {
            AppendLog("加载抽帧图片失败: " + ex.Message);
        }
    }

    private void RoiCanvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (_roiBitmap is null) { AppendLog("请先选择一张抽帧图片。"); return; }
        var id = RoiIdTextBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(id)) { AppendLog("请先填写 ROI ID。"); return; }
        var point = ClampToImage(e.GetPosition(RoiOverlayCanvas), requireInside: true);
        if (point is null) return;

        _roiDragStart = point;
        _roiDraftRectangle = CreateRoiRectangle(Brushes.DeepSkyBlue, 2, 0.18);
        RoiOverlayCanvas.Children.Add(_roiDraftRectangle);
        Canvas.SetLeft(_roiDraftRectangle, point.Value.X);
        Canvas.SetTop(_roiDraftRectangle, point.Value.Y);
        RoiOverlayCanvas.CaptureMouse();
        e.Handled = true;
    }

    private void RoiCanvas_MouseMove(object sender, MouseEventArgs e)
    {
        if (_roiDragStart is null || _roiDraftRectangle is null || e.LeftButton != MouseButtonState.Pressed) return;
        var point = ClampToImage(e.GetPosition(RoiOverlayCanvas));
        if (point is null) return;
        SetRectangleBounds(_roiDraftRectangle, _roiDragStart.Value, point.Value);
    }

    private void RoiCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (_roiDragStart is null || _roiDraftRectangle is null) return;
        var end = ClampToImage(e.GetPosition(RoiOverlayCanvas));
        RoiOverlayCanvas.ReleaseMouseCapture();
        var start = _roiDragStart.Value;
        _roiDragStart = null;
        _roiDraftRectangle = null;
        if (end is null || Math.Abs(end.Value.X - start.X) < 4 || Math.Abs(end.Value.Y - start.Y) < 4)
        {
            RefreshRoiEditor();
            AppendLog("ROI 区域过小，请重新拖动框选。");
            return;
        }
        var dialog = new RoiNameDialog(RoiIdTextBox.Text.Trim(), RoiNameTextBox.Text.Trim()) { Owner = this };
        if (dialog.ShowDialog() != true)
        {
            RefreshRoiEditor();
            AppendLog("已取消本次 ROI 圈选。");
            return;
        }
        RoiIdTextBox.Text = dialog.RoiId;
        RoiNameTextBox.Text = dialog.RoiName;
        UpsertRoiFromCanvas(start, end.Value, dialog.RoiId, dialog.RoiName);
    }

    private void UpsertRoiFromCanvas(Point start, Point end, string id, string name)
    {
        var imageRect = GetRenderedImageRect();
        if (imageRect.Width <= 0 || imageRect.Height <= 0) return;
        double NormalizeX(double value) => Math.Round(Math.Clamp((value - imageRect.X) / imageRect.Width, 0, 1), 6);
        double NormalizeY(double value) => Math.Round(Math.Clamp((value - imageRect.Y) / imageRect.Height, 0, 1), 6);
        var x1 = NormalizeX(Math.Min(start.X, end.X));
        var y1 = NormalizeY(Math.Min(start.Y, end.Y));
        var x2 = NormalizeX(Math.Max(start.X, end.X));
        var y2 = NormalizeY(Math.Max(start.Y, end.Y));

        var root = ParseObject(RoisJsonTextBox.Text, "ROI JSON") ?? new JsonObject();
        root["schema_version"] = "1.0";
        root["coordinate_type"] = "normalized";
        var regions = root["regions"] as JsonArray;
        if (regions is null)
        {
            regions = new JsonArray();
            root["regions"] = regions;
        }
        var existing = regions.OfType<JsonObject>().FirstOrDefault(region => region["id"]?.ToString() == id);
        var regionNode = existing ?? new JsonObject();
        regionNode["id"] = id;
        regionNode["name"] = name;
        regionNode["shape"] = "rectangle";
        regionNode["points"] = new JsonArray
        {
            new JsonArray(JsonValue.Create(x1), JsonValue.Create(y1)),
            new JsonArray(JsonValue.Create(x2), JsonValue.Create(y2)),
        };
        if (existing is null) regions.Add(regionNode);
        RoisJsonTextBox.Text = root.ToJsonString(PrettyJson);
        RefreshRoiEditor();
        AppendLog($"ROI {id} 已生成归一化坐标 [{x1}, {y1}] - [{x2}, {y2}]。");
    }

    private void RefreshRoiCanvasButton_Click(object sender, RoutedEventArgs e) => RefreshRoiEditor();

    private void DeleteSelectedRoiButton_Click(object sender, RoutedEventArgs e)
    {
        if (RoiRegionsListBox.SelectedItem is not RoiListItem selected) return;
        var root = ParseObject(RoisJsonTextBox.Text, "ROI JSON");
        if (root?["regions"] is not JsonArray regions) return;
        var target = regions.OfType<JsonObject>().FirstOrDefault(region => region["id"]?.ToString() == selected.Id);
        if (target is not null) regions.Remove(target);
        RoisJsonTextBox.Text = root.ToJsonString(PrettyJson);
        RefreshRoiEditor();
    }

    private void RoiRegionsListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (RoiRegionsListBox.SelectedItem is not RoiListItem selected) return;
        RoiIdTextBox.Text = selected.Id;
        RoiNameTextBox.Text = selected.Name;
    }

    private void RoiImageHost_SizeChanged(object sender, SizeChangedEventArgs e) => RefreshRoiEditor();

    private void RefreshRoiEditor()
    {
        if (RoiOverlayCanvas is null || RoiRegionsListBox is null) return;
        RoiOverlayCanvas.Children.Clear();
        RoiRegionsListBox.Items.Clear();
        var root = ParseObject(RoisJsonTextBox.Text, "ROI JSON");
        if (root?["regions"] is not JsonArray regions) return;
        var coordinateType = root["coordinate_type"]?.ToString() ?? "normalized";
        var imageRect = GetRenderedImageRect();
        var index = 0;
        foreach (var region in regions.OfType<JsonObject>())
        {
            if (region["points"] is not JsonArray points || points.Count != 2 || points[0] is not JsonArray p1 || points[1] is not JsonArray p2) continue;
            var id = region["id"]?.ToString() ?? $"roi_{index + 1}";
            var name = region["name"]?.ToString() ?? id;
            var x1 = p1[0]?.GetValue<double>() ?? 0;
            var y1 = p1[1]?.GetValue<double>() ?? 0;
            var x2 = p2[0]?.GetValue<double>() ?? 0;
            var y2 = p2[1]?.GetValue<double>() ?? 0;
            RoiRegionsListBox.Items.Add(new RoiListItem(id, name, x1, y1, x2, y2));
            if (_roiBitmap is null || imageRect.Width <= 0) { index++; continue; }
            if (coordinateType == "pixel")
            {
                x1 /= _roiBitmap.PixelWidth; x2 /= _roiBitmap.PixelWidth;
                y1 /= _roiBitmap.PixelHeight; y2 /= _roiBitmap.PixelHeight;
            }
            var start = new Point(imageRect.X + x1 * imageRect.Width, imageRect.Y + y1 * imageRect.Height);
            var end = new Point(imageRect.X + x2 * imageRect.Width, imageRect.Y + y2 * imageRect.Height);
            var rectangle = CreateRoiRectangle(RoiColors[index % RoiColors.Length], 2, 0.12);
            rectangle.ToolTip = $"{id} · {name}";
            SetRectangleBounds(rectangle, start, end);
            RoiOverlayCanvas.Children.Add(rectangle);
            index++;
        }
        RefreshWorkflowChoices();
    }

    private Rect GetRenderedImageRect()
    {
        if (_roiBitmap is null || RoiImageHost.ActualWidth <= 0 || RoiImageHost.ActualHeight <= 0) return Rect.Empty;
        var scale = Math.Min(RoiImageHost.ActualWidth / _roiBitmap.PixelWidth, RoiImageHost.ActualHeight / _roiBitmap.PixelHeight);
        var width = _roiBitmap.PixelWidth * scale;
        var height = _roiBitmap.PixelHeight * scale;
        return new Rect((RoiImageHost.ActualWidth - width) / 2, (RoiImageHost.ActualHeight - height) / 2, width, height);
    }

    private Point? ClampToImage(Point point, bool requireInside = false)
    {
        var rect = GetRenderedImageRect();
        if (rect.IsEmpty || (requireInside && !rect.Contains(point))) return null;
        return new Point(Math.Clamp(point.X, rect.Left, rect.Right), Math.Clamp(point.Y, rect.Top, rect.Bottom));
    }

    private static Rectangle CreateRoiRectangle(Brush stroke, double thickness, double opacity) => new()
    {
        Stroke = stroke, StrokeThickness = thickness, Fill = Brushes.Transparent,
        Opacity = Math.Max(opacity, 0.75), IsHitTestVisible = false,
    };

    private static void SetRectangleBounds(Rectangle rectangle, Point start, Point end)
    {
        Canvas.SetLeft(rectangle, Math.Min(start.X, end.X));
        Canvas.SetTop(rectangle, Math.Min(start.Y, end.Y));
        rectangle.Width = Math.Abs(end.X - start.X);
        rectangle.Height = Math.Abs(end.Y - start.Y);
    }

    private void LoadWorkflowSample()
    {
        var className = ParseClasses().FirstOrDefault()?.name ?? "part_a";
        WorkflowJsonTextBox.Text = FormatJson(new
        {
            schema_version = "1.0",
            workflow_id = $"{NewProjectIdTextBox.Text.Trim()}_V1",
            name = NewProjectNameTextBox.Text.Trim(),
            version = "1.0.0",
            steps = new[]
            {
                new { id = "step_001", order = 1, name = "放置零件", required = true, trigger = new { type = "object_in_roi", class_name = className, roi_id = "work", confidence = 0.4, stable_frames = 3 } },
            },
        });
    }

    private void AddWorkflowStepButton_Click(object sender, RoutedEventArgs e)
    {
        RefreshWorkflowChoices();
        var order = WorkflowSteps.Count + 1;
        var item = new WorkflowStepItem
        {
            Order = order,
            Id = $"step_{order:000}",
            Name = $"步骤 {order}",
            TriggerType = "object_in_roi",
            ClassName = AvailableClassNames.FirstOrDefault() ?? string.Empty,
            RoiId = AvailableRoiIds.FirstOrDefault() ?? string.Empty,
            Confidence = 0.25,
            StableFrames = 3,
            Required = true,
        };
        WorkflowSteps.Add(item);
        WorkflowStepsDataGrid.SelectedItem = item;
        WorkflowStepsDataGrid.ScrollIntoView(item);
    }

    private void DeleteWorkflowStepButton_Click(object sender, RoutedEventArgs e)
    {
        if (WorkflowStepsDataGrid.SelectedItem is not WorkflowStepItem selected) return;
        WorkflowSteps.Remove(selected);
        ResequenceWorkflowSteps();
    }

    private void MoveWorkflowStepUpButton_Click(object sender, RoutedEventArgs e) => MoveSelectedWorkflowStep(-1);
    private void MoveWorkflowStepDownButton_Click(object sender, RoutedEventArgs e) => MoveSelectedWorkflowStep(1);

    private void MoveSelectedWorkflowStep(int offset)
    {
        if (WorkflowStepsDataGrid.SelectedItem is not WorkflowStepItem selected) return;
        var current = WorkflowSteps.IndexOf(selected);
        var target = current + offset;
        if (target < 0 || target >= WorkflowSteps.Count) return;
        WorkflowSteps.Move(current, target);
        ResequenceWorkflowSteps();
        WorkflowStepsDataGrid.SelectedItem = selected;
    }

    private void ResequenceWorkflowSteps()
    {
        for (var index = 0; index < WorkflowSteps.Count; index++) WorkflowSteps[index].Order = index + 1;
    }

    private void RefreshWorkflowChoices()
    {
        var classes = new List<string>();
        var projectPath = Path.Combine(GetProjectDir(), "project.json");
        try
        {
            if (File.Exists(projectPath) && JsonNode.Parse(File.ReadAllText(projectPath)) is JsonObject project && project["classes"] is JsonArray projectClasses)
            {
                classes.AddRange(projectClasses.OfType<JsonObject>().Select(item => item["name"]?.ToString()).OfType<string>());
            }
        }
        catch (Exception ex) { AppendLog("读取项目类别失败: " + ex.Message); }
        if (classes.Count == 0)
        {
            classes.AddRange(Lines(ClassesTextBox.Text).Select(line => line.Split('|', 2, StringSplitOptions.TrimEntries)[0]));
        }
        ReplaceCollection(AvailableClassNames, classes.Distinct());

        var roiIds = new List<string>();
        var roiRoot = ParseObject(RoisJsonTextBox.Text, "ROI JSON");
        if (roiRoot?["regions"] is JsonArray regions)
        {
            roiIds.AddRange(regions.OfType<JsonObject>().Select(item => item["id"]?.ToString()).OfType<string>());
        }
        ReplaceCollection(AvailableRoiIds, roiIds.Distinct());
    }

    private static void ReplaceCollection(ObservableCollection<string> collection, IEnumerable<string> values)
    {
        collection.Clear();
        foreach (var value in values.Where(value => !string.IsNullOrWhiteSpace(value))) collection.Add(value);
    }

    private void LoadWorkflowEditorFromJson()
    {
        RefreshWorkflowChoices();
        var root = ParseObject(WorkflowJsonTextBox.Text, "流程 JSON");
        if (root?["steps"] is not JsonArray steps) return;
        WorkflowSteps.Clear();
        foreach (var node in steps.OfType<JsonObject>().OrderBy(item => item["order"]?.GetValue<int>() ?? int.MaxValue))
        {
            if (node["trigger"] is not JsonObject trigger) continue;
            WorkflowSteps.Add(new WorkflowStepItem
            {
                Order = node["order"]?.GetValue<int>() ?? WorkflowSteps.Count + 1,
                Id = node["id"]?.ToString() ?? string.Empty,
                Name = node["name"]?.ToString() ?? string.Empty,
                Required = node["required"]?.GetValue<bool>() ?? true,
                TriggerType = trigger["type"]?.ToString() ?? "object_in_roi",
                EventType = trigger["event"]?.ToString()
                    ?? (trigger["require_transition"]?.GetValue<bool>() == true ? "object_enter_roi" : string.Empty),
                ClassName = trigger["class_name"]?.ToString() ?? string.Empty,
                RoiId = trigger["roi_id"]?.ToString() ?? string.Empty,
                Confidence = trigger["confidence"]?.GetValue<double>() ?? 0.25,
                StableFrames = trigger["stable_frames"]?.GetValue<int>() ?? 3,
                AllowHandPose = trigger["allow_hand_pose"]?.GetValue<bool>() ?? false,
                TimeoutSec = node["timeout_sec"]?.ToString() ?? string.Empty,
                AdvancedTriggerJson = trigger.ToJsonString(),
            });
        }
        ResequenceWorkflowSteps();
    }

    private JsonObject BuildWorkflowJsonFromEditor()
    {
        if (WorkflowSteps.Count == 0) throw new InvalidOperationException("至少需要定义一个流程步骤。");
        var root = ParseObject(WorkflowJsonTextBox.Text, "流程 JSON") ?? new JsonObject();
        root["schema_version"] = "1.0";
        root["workflow_id"] ??= $"{CurrentProjectText.Text}_V1";
        root["name"] ??= $"{CurrentProjectText.Text} 流程";
        root["version"] ??= "1.0.0";
        var ids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var steps = new JsonArray();
        ResequenceWorkflowSteps();
        foreach (var item in WorkflowSteps)
        {
            var advanced = item.TriggerType is "composite" or "object_count"
                or "object_transition" or "duration" or "object_event";
            if (string.IsNullOrWhiteSpace(item.Id) || !ids.Add(item.Id)) throw new InvalidOperationException($"步骤 ID 不能为空或重复: {item.Id}");
            if (string.IsNullOrWhiteSpace(item.Name)) throw new InvalidOperationException($"步骤 {item.Id} 缺少名称。");
            if (!TriggerTypes.Contains(item.TriggerType)) throw new InvalidOperationException($"步骤 {item.Id} 的触发方式无效。");
            if (!EventTypes.Contains(item.EventType)) throw new InvalidOperationException($"步骤 {item.Id} 的对象事件无效。");
            if (item.Confidence is < 0 or > 1) throw new InvalidOperationException($"步骤 {item.Id} 的置信度必须位于 0～1。");
            if (item.StableFrames <= 0) throw new InvalidOperationException($"步骤 {item.Id} 的稳定帧必须大于 0。");
            if (!advanced && item.TriggerType != "hand_in_roi" && string.IsNullOrWhiteSpace(item.ClassName)) throw new InvalidOperationException($"步骤 {item.Id} 必须选择识别类别。");
            if (!advanced && item.TriggerType != "object_present" && string.IsNullOrWhiteSpace(item.RoiId)) throw new InvalidOperationException($"步骤 {item.Id} 必须选择 ROI。");

            JsonObject trigger;
            if (advanced)
            {
                trigger = ParseObject(item.AdvancedTriggerJson, $"步骤 {item.Id} 高级触发 JSON")
                    ?? throw new InvalidOperationException($"步骤 {item.Id} 必须填写高级触发 JSON。");
                trigger["type"] = item.TriggerType;
            }
            else
            {
                trigger = new JsonObject
                {
                    ["type"] = item.TriggerType,
                    ["confidence"] = item.Confidence,
                    ["stable_frames"] = item.StableFrames,
                };
                if (item.TriggerType != "hand_in_roi") trigger["class_name"] = item.ClassName;
                if (item.TriggerType != "object_present") trigger["roi_id"] = item.RoiId;
                if (item.AllowHandPose) trigger["allow_hand_pose"] = true;
                if (!string.IsNullOrWhiteSpace(item.EventType)) trigger["event"] = item.EventType;
            }
            var step = new JsonObject
            {
                ["id"] = item.Id,
                ["order"] = item.Order,
                ["name"] = item.Name,
                ["required"] = item.Required,
                ["trigger"] = trigger,
            };
            if (!string.IsNullOrWhiteSpace(item.TimeoutSec))
            {
                if (!double.TryParse(item.TimeoutSec, out var timeout) || timeout <= 0) throw new InvalidOperationException($"步骤 {item.Id} 的超时秒数必须大于 0。");
                step["timeout_sec"] = timeout;
            }
            steps.Add(step);
        }
        root["steps"] = steps;
        return root;
    }

    private async Task<JsonObject?> SendCommandAsync(object request, bool renameParams = false)
    {
        SetBusy(true);
        try
        {
            JsonNode? requestNode = request as JsonNode ?? JsonSerializer.SerializeToNode(request);
            if (renameParams && requestNode is JsonObject renameRoot && renameRoot.Remove("params_", out var paramsNode))
            {
                renameRoot["params"] = paramsNode;
            }
            NormalizeRequestPaths(requestNode);
            var text = requestNode?.ToJsonString(PrettyJson) ?? "{}";
            AppendLog(">>> " + text);

            using var client = new TcpClient();
            using var connectCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            await client.ConnectAsync(HostTextBox.Text.Trim(), GetPort(), connectCts.Token);
            await using var stream = client.GetStream();
            var requestBytes = Encoding.UTF8.GetBytes(text);
            await stream.WriteAsync(requestBytes);
            await stream.FlushAsync();
            var responseText = await ReadResponseAsync(stream);
            var response = JsonNode.Parse(responseText) as JsonObject;
            AppendLog("<<< " + (response?.ToJsonString(PrettyJson) ?? responseText));
            var ok = response?["status"]?.GetValue<string>() == "ok";
            StatusTextBlock.Text = ok ? "请求完成" : response?["message"]?.ToString() ?? "请求失败";
            return response;
        }
        catch (OperationCanceledException)
        {
            AppendLog("请求超时，请确认算法服务正在运行。长时间训练最多等待 12 小时。");
            StatusTextBlock.Text = "请求超时";
            return null;
        }
        catch (Exception ex)
        {
            AppendLog("请求失败: " + ex.Message);
            StatusTextBlock.Text = "请求失败";
            return null;
        }
        finally
        {
            SetBusy(false);
        }
    }

    private static async Task<string> ReadResponseAsync(NetworkStream stream)
    {
        using var readCts = new CancellationTokenSource(TimeSpan.FromHours(12));
        using var memory = new MemoryStream();
        var buffer = new byte[256 * 1024];
        var count = await stream.ReadAsync(buffer.AsMemory(), readCts.Token);
        if (count == 0) return "{}";
        memory.Write(buffer, 0, count);
        await Task.Delay(20, readCts.Token);
        while (stream.DataAvailable)
        {
            count = await stream.ReadAsync(buffer.AsMemory(), readCts.Token);
            if (count == 0) break;
            memory.Write(buffer, 0, count);
            await Task.Delay(10, readCts.Token);
        }
        return Encoding.UTF8.GetString(memory.ToArray());
    }

    private void NormalizeRequestPaths(JsonNode? request)
    {
        if (request is not JsonObject root || root["params"] is not JsonObject parameters) return;
        var command = root["command"]?.ToString();
        foreach (var key in PathParameterNames)
        {
            // detect.output_dir 是当前 SOP 项目内部的相对路径，由后端安全解析。
            if (key == "output_dir" && command == "detect") continue;
            if (parameters[key] is JsonValue valueNode && valueNode.TryGetValue<string>(out var value))
            {
                parameters[key] = NormalizePackagePath(value);
            }
        }
        if (parameters["video_paths"] is JsonArray paths)
        {
            for (var index = 0; index < paths.Count; index++)
            {
                if (paths[index] is JsonValue item && item.TryGetValue<string>(out var value))
                {
                    paths[index] = NormalizePackagePath(value);
                }
            }
        }
    }

    private string NormalizePackagePath(string value)
    {
        if (string.IsNullOrWhiteSpace(value) || Path.IsPathRooted(value)) return value;
        return Path.GetFullPath(Path.Combine(GetPackageRoot(), value));
    }

    private object ProjectCommand(string command) => new { command, @params = new { sop_project_dir = GetProjectDir() } };

    private void UpdateProjectSummary(JsonObject? response, bool loadEditors = false)
    {
        var data = GetData(response);
        if (data is null) return;
        var project = data["project"] as JsonObject;
        var projectDir = data["project_dir"]?.ToString();
        if (!string.IsNullOrWhiteSpace(projectDir)) ProjectDirTextBox.Text = projectDir;
        CurrentProjectText.Text = project?["project_id"]?.ToString() ?? "未知项目";
        var status = project?["status"]?.ToString() ?? "draft";
        var ready = data["ready"]?.GetValue<bool>() ?? false;
        ProjectStateText.Text = $"状态：{status} · 配置{(ready ? "完整" : "未完成")}";
        RefreshModelChoices();
        RefreshConversionModelChoices();
        RefreshDatasetChoices();
        if (loadEditors)
        {
            if (data["rois"] is JsonNode rois) RoisJsonTextBox.Text = rois.ToJsonString(PrettyJson);
            if (data["workflow"] is JsonNode workflow) WorkflowJsonTextBox.Text = workflow.ToJsonString(PrettyJson);
            RefreshRoiEditor();
            LoadWorkflowEditorFromJson();
        }
    }

    private static JsonObject? GetData(JsonObject? response) => response?["data"] as JsonObject;

    private static bool HasCapability(JsonObject? response, string capability)
    {
        var capabilities = GetData(response)?["capabilities"] as JsonArray;
        return capabilities?.Any(item => string.Equals(item?.ToString(), capability, StringComparison.Ordinal)) == true;
    }

    private List<ClassDefinition> ParseClasses()
    {
        var result = new List<ClassDefinition>();
        foreach (var line in Lines(ClassesTextBox.Text))
        {
            var parts = line.Split('|', 2, StringSplitOptions.TrimEntries);
            result.Add(new ClassDefinition(result.Count, parts[0], parts.Length > 1 ? parts[1] : parts[0]));
        }
        if (result.Count == 0) throw new InvalidOperationException("至少需要定义一个识别类别。");
        return result;
    }

    private static IEnumerable<string> Lines(string text) => text.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    private static string FormatJson(object value) => JsonSerializer.Serialize(value, PrettyJson);

    private JsonObject? ParseObject(string text, string name)
    {
        try { return JsonNode.Parse(text) as JsonObject ?? throw new JsonException("根节点必须是对象"); }
        catch (Exception ex) { AppendLog($"{name} 格式错误: {ex.Message}"); return null; }
    }

    private static int ParseInt(TextBox textBox, string name) => int.TryParse(textBox.Text.Trim(), out var value) ? value : throw new InvalidOperationException($"{name} 必须是整数。");
    private static double ParseDouble(TextBox textBox, string name) => double.TryParse(textBox.Text.Trim(), out var value) ? value : throw new InvalidOperationException($"{name} 必须是数字。");

    private static void SetOptionalText(JsonObject target, string key, TextBox textBox)
    {
        var value = textBox.Text.Trim();
        if (!string.IsNullOrWhiteSpace(value)) target[key] = value;
    }

    private static void SetOptionalInt(JsonObject target, string key, TextBox textBox, string displayName)
    {
        var text = textBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(text)) return;
        if (!int.TryParse(text, out var value)) throw new InvalidOperationException($"{displayName} 必须是整数。");
        target[key] = value;
    }

    private static void SetOptionalDouble(JsonObject target, string key, TextBox textBox)
    {
        var text = textBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(text)) return;
        if (!double.TryParse(text, out var value)) throw new InvalidOperationException($"{key} 必须是数字。");
        target[key] = value;
    }

    private static void SetOptionalJsonObject(JsonObject target, string key, TextBox textBox)
    {
        var text = textBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(text)) return;
        try
        {
            target[key] = JsonNode.Parse(text) as JsonObject
                ?? throw new JsonException("根节点必须是 JSON 对象");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"{key} 格式错误: {ex.Message}", ex);
        }
    }

    private void ProjectsListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (ProjectsListBox.SelectedItem is ProjectListItem item)
        {
            ProjectDirTextBox.Text = item.ProjectDir;
            CurrentProjectText.Text = item.Id;
            ProjectStateText.Text = $"状态：{item.Status} · 配置{(item.Ready ? "完整" : "未完成")}";
            RefreshModelChoices();
        }
    }

    private void BrowseRootButton_Click(object sender, RoutedEventArgs e) => BrowseFolderInto(PackageRootTextBox);
    private void BrowseProjectButton_Click(object sender, RoutedEventArgs e) => BrowseFolderInto(ProjectDirTextBox);
    private void BrowseLabelsButton_Click(object sender, RoutedEventArgs e) => BrowseFolderInto(LabelsDirTextBox);
    private void BrowseVideosButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog { Filter = "视频文件|*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.m4v|所有文件|*.*", Multiselect = true };
        if (dialog.ShowDialog() == true) VideoPathsTextBox.Text = string.Join(Environment.NewLine, dialog.FileNames);
    }
    private void BrowseDetectVideoButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog { Filter = "视频文件|*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.m4v|所有文件|*.*" };
        if (dialog.ShowDialog() == true) DetectVideoTextBox.Text = dialog.FileName;
    }

    private void RefreshModelsButton_Click(object sender, RoutedEventArgs e) => RefreshModelChoices();

    private void RefreshConversionModelsButton_Click(object sender, RoutedEventArgs e) => RefreshConversionModelChoices();

    private void RefreshDatasetChoices()
    {
        if (DatasetNameComboBox is null) return;
        var previous = DatasetNameComboBox.Text.Trim();
        var datasetRoot = Path.Combine(GetProjectDir(), "dataset");
        var datasets = new List<string>();
        if (Directory.Exists(datasetRoot))
        {
            datasets.AddRange(
                Directory.EnumerateDirectories(datasetRoot)
                    .Where(path => File.Exists(Path.Combine(path, "data.yaml"))
                        && File.Exists(Path.Combine(path, "dataset_manifest.json")))
                    .Select(Path.GetFileName)
                    .OfType<string>()
                    .OrderBy(name => name, StringComparer.OrdinalIgnoreCase)
            );
        }
        DatasetNameComboBox.ItemsSource = datasets;
        DatasetNameComboBox.Text = datasets.Contains(previous, StringComparer.OrdinalIgnoreCase)
            ? previous
            : datasets.FirstOrDefault() ?? (string.IsNullOrWhiteSpace(previous) ? "dataset_v1" : previous);
    }

    private void RefreshConversionModelChoices()
    {
        if (ConversionModelComboBox is null) return;
        var previous = (ConversionModelComboBox.SelectedItem as ModelVersionChoice)?.Key;
        var choices = new List<ModelVersionChoice>();
        var modelsRoot = Path.Combine(GetProjectDir(), "models");
        if (Directory.Exists(modelsRoot))
        {
            foreach (var manifestPath in Directory.EnumerateFiles(modelsRoot, "model_manifest.json", SearchOption.AllDirectories))
            {
                try
                {
                    if (JsonNode.Parse(File.ReadAllText(manifestPath)) is not JsonObject manifest) continue;
                    var modelId = manifest["model_id"]?.ToString();
                    var modelVersion = manifest["model_version"]?.ToString();
                    var bestPt = Path.Combine(Path.GetDirectoryName(manifestPath)!, "best.pt");
                    if (!string.IsNullOrWhiteSpace(modelId) && !string.IsNullOrWhiteSpace(modelVersion) && File.Exists(bestPt))
                    {
                        choices.Add(new ModelVersionChoice(modelId, modelVersion));
                    }
                }
                catch (JsonException)
                {
                    // Ignore invalid manifests; the backend will continue to validate selected models.
                }
            }
        }

        ConversionModelComboBox.ItemsSource = choices
            .OrderBy(choice => choice.ModelId, StringComparer.OrdinalIgnoreCase)
            .ThenByDescending(choice => choice.ModelVersion, StringComparer.OrdinalIgnoreCase)
            .ToList();
        ConversionModelComboBox.SelectedItem = choices.FirstOrDefault(choice => choice.Key == previous)
            ?? ConversionModelComboBox.Items.Cast<ModelVersionChoice>().FirstOrDefault();
    }

    private void BrowseDetectModelButton_Click(object sender, RoutedEventArgs e)
    {
        var projectModels = Path.Combine(GetProjectDir(), "models");
        var dialog = new OpenFileDialog
        {
            Title = "选择检测模型",
            Filter = "支持的模型|*.pt;*.onnx;*.engine;*.torchscript|PyTorch 模型|*.pt|ONNX 模型|*.onnx|TensorRT 模型|*.engine|所有文件|*.*",
            InitialDirectory = Directory.Exists(projectModels) ? projectModels : GetProjectDir(),
        };
        if (dialog.ShowDialog() != true) return;
        if (!DetectModelComboBox.Items.Cast<object>().Any(item => string.Equals(item?.ToString(), dialog.FileName, StringComparison.OrdinalIgnoreCase)))
        {
            DetectModelComboBox.Items.Insert(0, dialog.FileName);
        }
        DetectModelComboBox.SelectedItem = dialog.FileName;
    }

    private void RefreshModelChoices()
    {
        if (DetectModelComboBox is null) return;
        var previous = DetectModelComboBox.SelectedItem?.ToString()?.Trim() ?? string.Empty;
        var candidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string? activeModelPath = null;
        var projectDir = GetProjectDir();
        var projectModels = Path.Combine(projectDir, "models");
        AddModelsFromDirectory(projectModels, candidates);

        var projectJson = Path.Combine(projectDir, "project.json");
        if (File.Exists(projectJson))
        {
            try
            {
                var project = JsonNode.Parse(File.ReadAllText(projectJson)) as JsonObject;
                var activeModel = project?["active_model"]?.ToString();
                if (!string.IsNullOrWhiteSpace(activeModel))
                {
                    var resolved = Path.IsPathRooted(activeModel) ? activeModel : Path.GetFullPath(Path.Combine(projectDir, activeModel));
                    if (File.Exists(resolved))
                    {
                        activeModelPath = resolved;
                        candidates.Add(resolved);
                    }
                }
            }
            catch (Exception ex)
            {
                AppendLog("读取项目活动模型失败: " + ex.Message);
            }
        }

        DetectModelComboBox.Items.Clear();
        foreach (var path in candidates.OrderByDescending(File.GetLastWriteTimeUtc).ThenBy(path => path))
        {
            DetectModelComboBox.Items.Add(path);
        }
        if (!string.IsNullOrWhiteSpace(previous) && candidates.Contains(previous))
        {
            DetectModelComboBox.SelectedItem = previous;
        }
        else if (!string.IsNullOrWhiteSpace(activeModelPath))
        {
            DetectModelComboBox.SelectedItem = activeModelPath;
        }
        else if (DetectModelComboBox.Items.Count > 0)
        {
            DetectModelComboBox.SelectedIndex = 0;
        }
        AppendLog($"已发现 {DetectModelComboBox.Items.Count} 个当前项目可用模型。");
    }

    private static void AddModelsFromDirectory(string directory, HashSet<string> target)
    {
        if (!Directory.Exists(directory)) return;
        var supported = new HashSet<string>([".pt", ".onnx", ".engine", ".torchscript"], StringComparer.OrdinalIgnoreCase);
        foreach (var path in Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories))
        {
            if (supported.Contains(Path.GetExtension(path))) target.Add(Path.GetFullPath(path));
        }
    }
    private static void BrowseFolderInto(TextBox target)
    {
        var dialog = new OpenFolderDialog { InitialDirectory = Directory.Exists(target.Text) ? target.Text : null };
        if (dialog.ShowDialog() == true) target.Text = dialog.FolderName;
    }

    private async void StartServerButton_Click(object sender, RoutedEventArgs e)
    {
        if (_serverProcess is { HasExited: false }) { AppendLog("本地算法服务已在运行。"); return; }
        var root = GetPackageRoot();
        var exePath = Path.Combine(root, "SOP_PYD.exe");
        var pythonPath = Path.Combine(root, ".venv", "Scripts", "python.exe");
        var scriptPath = Path.Combine(root, "SOP_PYD.py");
        ProcessStartInfo startInfo;
        if (File.Exists(exePath))
        {
            startInfo = BuildStartInfo(exePath, $"tcp {GetPort()}", root);
        }
        else if (File.Exists(pythonPath) && File.Exists(scriptPath))
        {
            startInfo = BuildStartInfo(pythonPath, $"\"{scriptPath}\" tcp {GetPort()}", root);
        }
        else
        {
            AppendLog("未找到 SOP_PYD.exe，也未找到源码虚拟环境入口。");
            return;
        }

        try
        {
            _serverProcess = Process.Start(startInfo);
            if (_serverProcess is null) throw new InvalidOperationException("进程启动失败");
            _serverProcess.OutputDataReceived += (_, args) => { if (!string.IsNullOrWhiteSpace(args.Data)) Dispatcher.Invoke(() => AppendLog("[server] " + args.Data)); };
            _serverProcess.ErrorDataReceived += (_, args) => { if (!string.IsNullOrWhiteSpace(args.Data)) Dispatcher.Invoke(() => AppendLog("[server:error] " + args.Data)); };
            _serverProcess.BeginOutputReadLine();
            _serverProcess.BeginErrorReadLine();
            AppendLog("本地算法服务进程已创建，正在确认端口监听……");
            await Task.Delay(800);
            if (_serverProcess.HasExited)
            {
                AppendLog($"本地算法服务启动失败，进程退出码: {_serverProcess.ExitCode}。请检查端口是否已被旧服务占用。");
                _serverProcess.Dispose();
                _serverProcess = null;
                return;
            }
            var health = await SendCommandAsync(new { command = "health" });
            if (HasCapability(health, "training_stop_after_epoch"))
            {
                AppendLog("新版本地算法服务已启动并通过能力检查。");
            }
            else
            {
                AppendLog("端口响应来自旧版算法服务，请关闭占用端口的旧进程后重试。");
            }
        }
        catch (Exception ex) { AppendLog("启动服务失败: " + ex.Message); }
    }

    private static ProcessStartInfo BuildStartInfo(string fileName, string arguments, string workingDirectory) => new()
    {
        FileName = fileName, Arguments = arguments, WorkingDirectory = workingDirectory,
        UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true,
        RedirectStandardError = true, StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
    };

    private async void CloseServerButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            using var client = new TcpClient();
            using var connectCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            await client.ConnectAsync(HostTextBox.Text.Trim(), GetPort(), connectCts.Token);
            await using var stream = client.GetStream();
            await stream.WriteAsync(Encoding.UTF8.GetBytes("close"));
            await stream.FlushAsync();
            AppendLog("已发送停止服务指令。");
        }
        catch (Exception ex)
        {
            AppendLog("停止服务失败: " + ex.Message);
        }
        ServiceIndicator.Fill = new SolidColorBrush(Color.FromRgb(148, 163, 184));
        ServiceStatusText.Text = "未连接";
    }

    private void ClearButton_Click(object sender, RoutedEventArgs e) { ResponseTextBox.Clear(); StatusTextBlock.Text = "就绪"; }
    private int GetPort() => int.TryParse(PortTextBox.Text.Trim(), out var port) && port is > 0 and <= 65535 ? port : throw new InvalidOperationException("端口必须在 1 到 65535 之间。");
    private string GetPackageRoot() => PackageRootTextBox.Text.Trim();
    private string GetProjectDir() => ProjectDirTextBox.Text.Trim();

    private void SetBusy(bool busy)
    {
        WorkflowTabs.IsEnabled = !busy || _trainingActive;
        HealthButton.IsEnabled = !busy;
        StatusTextBlock.Text = busy ? "正在发送请求…" : StatusTextBlock.Text;
    }

    private void AppendLog(string message)
    {
        ResponseTextBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}{Environment.NewLine}");
        ResponseTextBox.ScrollToEnd();
    }

    private static string FindPackageRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(dir))
        {
            if (File.Exists(Path.Combine(dir, "SOP_PYD.py")) || File.Exists(Path.Combine(dir, "SOP_PYD.exe"))) return dir;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return @"E:\Project\SOPAID\SOPAID";
    }

    protected override void OnClosed(EventArgs e) { base.OnClosed(e); _serverProcess?.Dispose(); }

    private static readonly string[] PathParameterNames =
    [
        "sop_project_dir", "video_path", "model_path", "output_dir", "hand_pose_model_path",
        "base_model_path", "labels_dir", "data_yaml_path", "project_dir", "images_dir",
        "dataset_output_dir", "copy_best_to", "model", "data",
    ];

    private sealed record ClassDefinition(int id, string name, string display_name);
    private sealed record ModelVersionChoice(string ModelId, string ModelVersion)
    {
        public string Key => $"{ModelId}/{ModelVersion}";
        public override string ToString() => $"{ModelId} / {ModelVersion}";
    }
    public sealed class WorkflowStepItem : INotifyPropertyChanged
    {
        private int _order;
        private string _id = string.Empty;
        private string _name = string.Empty;
        private string _triggerType = "object_in_roi";
        private string _className = string.Empty;
        private string _roiId = string.Empty;
        private string _eventType = string.Empty;
        private double _confidence = 0.25;
        private int _stableFrames = 3;
        private string _timeoutSec = string.Empty;
        private bool _required = true;
        private bool _allowHandPose;
        private string _advancedTriggerJson = string.Empty;

        public int Order { get => _order; set => SetField(ref _order, value); }
        public string Id { get => _id; set => SetField(ref _id, value); }
        public string Name { get => _name; set => SetField(ref _name, value); }
        public string TriggerType { get => _triggerType; set => SetField(ref _triggerType, value); }
        public string ClassName { get => _className; set => SetField(ref _className, value); }
        public string RoiId { get => _roiId; set => SetField(ref _roiId, value); }
        public string EventType { get => _eventType; set => SetField(ref _eventType, value); }
        public double Confidence { get => _confidence; set => SetField(ref _confidence, value); }
        public int StableFrames { get => _stableFrames; set => SetField(ref _stableFrames, value); }
        public string TimeoutSec { get => _timeoutSec; set => SetField(ref _timeoutSec, value); }
        public bool Required { get => _required; set => SetField(ref _required, value); }
        public bool AllowHandPose { get => _allowHandPose; set => SetField(ref _allowHandPose, value); }
        public string AdvancedTriggerJson { get => _advancedTriggerJson; set => SetField(ref _advancedTriggerJson, value); }

        public event PropertyChangedEventHandler? PropertyChanged;

        private void SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return;
            field = value;
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
    private sealed record RoiListItem(string Id, string Name, double X1, double Y1, double X2, double Y2)
    {
        public override string ToString() => $"{Id} · {Name}  ({X1:0.###},{Y1:0.###})-({X2:0.###},{Y2:0.###})";
    }
    private sealed record ProjectListItem(string Id, string Name, string Status, bool Ready, string ProjectDir)
    {
        public string DisplayName => $"{Id} · {Name} · {Status}{(Ready ? " ✓" : "")}";
    }
}
