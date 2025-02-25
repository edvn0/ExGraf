using System.Reflection;
using MetricsSubscriber.Configuration;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var host = Host.CreateDefaultBuilder()
	.ConfigureServices((context, services) =>
	{
		var builder = new ConfigurationBuilder()
			.SetBasePath(context.HostingEnvironment.ContentRootPath)
			.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
			.AddJsonFile($"appsettings.{context.HostingEnvironment.EnvironmentName}.json", optional: true)
			.AddEnvironmentVariables();
		var configuration = builder.Build();

		services.AddSingleton(configuration);
		services.AddLogging();

		services.AddMediatR(cfg => cfg.RegisterServicesFromAssembly(Assembly.GetExecutingAssembly()));
		services.AddMassTransit();

	})
	.Build();

try
{
	var env = host.Services.GetRequiredService<IHostEnvironment>();
	if (env.IsDevelopment())
	{
		using var scope = host.Services.CreateScope();
		var services = scope.ServiceProvider;
		var dbContext = services.GetRequiredService<ModelPerformanceContext>();
		await dbContext.Database.MigrateAsync();
	}

	await host.StartAsync();
	await host.WaitForShutdownAsync();
}
finally
{
	if (host is IAsyncDisposable disposableHost)
		await disposableHost.DisposeAsync();
	else
		host.Dispose();
}
