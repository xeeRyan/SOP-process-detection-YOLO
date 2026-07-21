using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows;

namespace SopAidTcpTester;

public partial class MainWindow : Window
{
    private Process? _serverProcess;

    public MainWindow()
    {
        InitializeComponent();
        PackageRootTextBox.Text = FindPackageRoot();
        LoadRequestTemplate("sample_health_request.json");
    }

    private async void HealthButton_Click(object sender, RoutedEventArgs e)
    {
        await SendTextAsync("""{"command":"health"}""");
    }

    private async void SendJsonButton_Click(object sender, RoutedEventArgs e)
    {
        await SendTextAsync(NormalizeRequestJsonPaths(RequestTextBox.Text));
    }

    private void LoadHealthButton_Click(object sender, RoutedEventArgs e)
    {
        LoadRequestTemplate("sample_health_request.json");
    }

    private void LoadDetectButton_Click(object sender, RoutedEventArgs e)
    {
        LoadRequestTemplate("sample_detect_request.json");
    }

    private void LoadTrainButton_Click(object sender, RoutedEventArgs e)
    {
        LoadRequestTemplate("sample_train_smoke_request.json");
    }

    private async void SendModelFileButton_Click(object sender, RoutedEventArgs e)
    {
        var jsonPath = ResolvePackagePath(JsonFileTextBox.Text.Trim());
        Directory.CreateDirectory(Path.GetDirectoryName(jsonPath) ?? GetPackageRoot());
        File.WriteAllText(jsonPath, NormalizeRequestJsonPaths(RequestTextBox.Text), Encoding.UTF8);
        AppendLog("Wrote package request file: " + jsonPath);
        await SendTextAsync($"model_file:{jsonPath}");
    }

    private async void CloseServerButton_Click(object sender, RoutedEventArgs e)
    {
        await SendTextAsync("close");
    }

    private void StartServerButton_Click(object sender, RoutedEventArgs e)
    {
        if (_serverProcess is { HasExited: false })
        {
            AppendLog("Local server is already running.");
            return;
        }

        var packageRoot = GetPackageRoot();
        var exePath = Path.Combine(packageRoot, "SOP_PYD.exe");
        if (!File.Exists(exePath))
        {
            AppendLog($"SOP_PYD.exe not found: {exePath}");
            return;
        }

        var port = GetPort();
        var startInfo = new ProcessStartInfo
        {
            FileName = exePath,
            Arguments = $"tcp {port}",
            WorkingDirectory = packageRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        try
        {
            _serverProcess = Process.Start(startInfo);
            if (_serverProcess is null)
            {
                AppendLog("Failed to start local server process.");
                return;
            }

            _serverProcess.OutputDataReceived += (_, args) =>
            {
                if (!string.IsNullOrWhiteSpace(args.Data))
                {
                    Dispatcher.Invoke(() => AppendLog("[server] " + args.Data));
                }
            };
            _serverProcess.ErrorDataReceived += (_, args) =>
            {
                if (!string.IsNullOrWhiteSpace(args.Data))
                {
                    Dispatcher.Invoke(() => AppendLog("[server:error] " + args.Data));
                }
            };
            _serverProcess.BeginOutputReadLine();
            _serverProcess.BeginErrorReadLine();
            AppendLog($"Started package exe on port {port}: {exePath}");
        }
        catch (Exception ex)
        {
            AppendLog("Start server failed: " + ex.Message);
        }
    }

    private void ClearButton_Click(object sender, RoutedEventArgs e)
    {
        ResponseTextBox.Clear();
        StatusTextBlock.Text = "Ready";
    }

    private async Task SendTextAsync(string text)
    {
        SetBusy(true);
        try
        {
            var host = HostTextBox.Text.Trim();
            var port = GetPort();
            using var client = new TcpClient();

            using var connectCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            await client.ConnectAsync(host, port, connectCts.Token);

            await using var stream = client.GetStream();
            var requestBytes = Encoding.UTF8.GetBytes(text.Trim());
            await stream.WriteAsync(requestBytes);
            await stream.FlushAsync();

            var response = await ReadResponseAsync(stream);
            AppendLog(">>> " + text.Trim());
            AppendLog("<<< " + response);
            StatusTextBlock.Text = "Last request succeeded.";
        }
        catch (OperationCanceledException)
        {
            AppendLog("Request timed out. Check whether SOP_PYD TCP service is running.");
            StatusTextBlock.Text = "Timed out.";
        }
        catch (Exception ex)
        {
            AppendLog("Request failed: " + ex.Message);
            StatusTextBlock.Text = "Failed.";
        }
        finally
        {
            SetBusy(false);
        }
    }

    private static async Task<string> ReadResponseAsync(NetworkStream stream)
    {
        var buffer = new byte[102400];
        using var readCts = new CancellationTokenSource(TimeSpan.FromHours(2));
        var length = await stream.ReadAsync(buffer.AsMemory(0, buffer.Length), readCts.Token);
        return length == 0 ? "<empty response>" : Encoding.UTF8.GetString(buffer, 0, length);
    }

    private void LoadRequestTemplate(string fileName)
    {
        var path = Path.Combine(AppContext.BaseDirectory, fileName);
        if (!File.Exists(path))
        {
            path = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", fileName));
        }

        RequestTextBox.Text = File.Exists(path)
            ? File.ReadAllText(path, Encoding.UTF8)
            : $$"""
              {
                "command": "health"
              }
              """;
        AppendLog("Loaded request template: " + fileName);
    }

    private string NormalizeRequestJsonPaths(string text)
    {
        try
        {
            var node = JsonNode.Parse(text);
            if (node is not JsonObject root)
            {
                return text;
            }

            var target = root["params"] as JsonObject ?? root;
            foreach (var key in PathParameterNames)
            {
                if (target[key] is JsonValue valueNode && valueNode.TryGetValue<string>(out var value))
                {
                    target[key] = NormalizePackagePathValue(key, value);
                }
            }

            return root.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
        }
        catch (JsonException)
        {
            return text;
        }
    }

    private string NormalizePackagePathValue(string key, string value)
    {
        if (string.IsNullOrWhiteSpace(value) || Path.IsPathRooted(value))
        {
            return value;
        }

        var packageRoot = GetPackageRoot();
        var packagePath = Path.GetFullPath(Path.Combine(packageRoot, value));
        if (IsInputPathParameter(key))
        {
            if (File.Exists(packagePath) || Directory.Exists(packagePath))
            {
                return packagePath;
            }

            var internalPath = Path.GetFullPath(Path.Combine(packageRoot, "_internal", value));
            if (File.Exists(internalPath) || Directory.Exists(internalPath))
            {
                return internalPath;
            }
        }

        return packagePath;
    }

    private int GetPort()
    {
        if (!int.TryParse(PortTextBox.Text.Trim(), out var port) || port <= 0 || port > 65535)
        {
            throw new InvalidOperationException("Port must be between 1 and 65535.");
        }

        return port;
    }

    private string ResolvePackagePath(string path)
    {
        if (Path.IsPathRooted(path))
        {
            return path;
        }

        return Path.GetFullPath(Path.Combine(GetPackageRoot(), path));
    }

    private static readonly string[] PathParameterNames =
    [
        "video_path",
        "model_path",
        "output_dir",
        "hand_pose_model_path",
        "base_model_path",
        "data_yaml_path",
        "project_dir",
        "images_dir",
        "labels_dir",
        "dataset_output_dir",
        "copy_best_to",
        "model",
        "data",
    ];

    private static bool IsInputPathParameter(string key)
    {
        return key is
            "video_path" or
            "model_path" or
            "hand_pose_model_path" or
            "base_model_path" or
            "data_yaml_path" or
            "images_dir" or
            "labels_dir" or
            "model" or
            "data";
    }

    private void SetBusy(bool busy)
    {
        HealthButton.IsEnabled = !busy;
        SendJsonButton.IsEnabled = !busy;
        SendModelFileButton.IsEnabled = !busy;
        CloseServerButton.IsEnabled = !busy;
        LoadHealthButton.IsEnabled = !busy;
        LoadDetectButton.IsEnabled = !busy;
        LoadTrainButton.IsEnabled = !busy;
        StatusTextBlock.Text = busy ? "Sending request..." : StatusTextBlock.Text;
    }

    private void AppendLog(string message)
    {
        ResponseTextBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}{Environment.NewLine}");
        ResponseTextBox.ScrollToEnd();
    }

    private string GetPackageRoot()
    {
        return PackageRootTextBox.Text.Trim();
    }

    private static string FindPackageRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(dir))
        {
            if (File.Exists(Path.Combine(dir, "SOP_PYD.exe")) && Directory.Exists(Path.Combine(dir, "_internal")))
            {
                return dir;
            }

            var parent = Directory.GetParent(dir);
            if (parent is null)
            {
                break;
            }

            dir = parent.FullName;
        }

        return @"E:\Project\SOPAID\SOPAID\dist\SOP_PYD";
    }

    protected override void OnClosed(EventArgs e)
    {
        base.OnClosed(e);
        _serverProcess?.Dispose();
    }
}
