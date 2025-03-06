namespace MetricsSubscriber.Models.Bus;

public record MetricsMessage
{
	public required int Epoch { get; init; }
	public required double Loss { get; init; }
	public required double Accuracy { get; init; }
	public required double MeanPPV { get; init; }
	public required double MeanFPR { get; init; }
	public required double MeanRecall { get; init; }
}
