using System.Text.RegularExpressions;
using System.Windows;

namespace SopAidTcpTester;

public partial class RoiNameDialog : Window
{
    private static readonly Regex ValidId = new("^[A-Za-z0-9_-]+$", RegexOptions.Compiled);

    public string RoiId => RoiIdInput.Text.Trim();
    public string RoiName => RoiNameInput.Text.Trim();

    public RoiNameDialog(string suggestedId, string suggestedName)
    {
        InitializeComponent();
        RoiIdInput.Text = suggestedId;
        RoiNameInput.Text = suggestedName;
        Loaded += (_, _) => { RoiIdInput.Focus(); RoiIdInput.SelectAll(); };
    }

    private void ConfirmButton_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(RoiId))
        {
            ValidationText.Text = "请填写 ROI ID";
            RoiIdInput.Focus();
            return;
        }
        if (!ValidId.IsMatch(RoiId))
        {
            ValidationText.Text = "ID 仅支持字母、数字、_ 和 -";
            RoiIdInput.Focus();
            return;
        }
        if (string.IsNullOrWhiteSpace(RoiName))
        {
            ValidationText.Text = "请填写显示名称";
            RoiNameInput.Focus();
            return;
        }
        DialogResult = true;
    }
}
